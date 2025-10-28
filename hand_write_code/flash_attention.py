# Parallelize each sequence in a batch, 
# inside each sequence we parallelize each head
# inside each head we parallelize each query block
# At most, we have BN X NUM_HEADS X (Block_SIZE_Q)

import triton
import torch
import triton.language as tl



@triton.jit
def _attn_fwd_inner(
    O_block,
    l_i,
    m_i, 
    Q_block,
    K_block_ptr,
    V_block_ptr,
    block_index_q,
    softmax_scale,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
    offs_q: tl.constexpr,
    offs_kv: tl.constexpr,
    SEQ_LEN: tl.constexpr,
):
    # range of values handled by this stage
    if STAGE == 1:
        # From 0 to the left of the diagonal
        # 从K的起始位置到Q的起始位置，对角线左侧的
        lo, hi = 0, block_index_q * BLOCK_SIZE_Q
    elif STAGE == 2:
        # Used only for the block in which there is transition between non-masked and masked keys
        # 从Q的起始位置到Q的结束位置，对角线的块。对角线的块里，有些是masked，有些是non-masked。
        lo, hi = block_index_q * BLOCK_SIZE_Q, (block_index_q + 1) * BLOCK_SIZE_Q
        lo = tl.multiple_of(lo, BLOCK_SIZE_Q) # assert lo % BLOCK_SIZE_Q == 0
    else:
        # Only used for the non-causal attention
        lo, hi = 0, SEQ_LEN
        
    K_block_ptr = tl.advance(K_block_ptr, (0, lo)) # K is transposed, so different from V
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))
    # New address = Old address + [(advance[0] * stride[0]), (advance[1] * stride[1])]
    
    # 现在K,V指向了起始位置 
    # loop k, v and upadate accumulator
    for start_kv in range(lo, hi, BLOCK_SIZE_KV):
        # Just let the compiler know that start_kv is a multiple of BLOCK_SIZE_KV, so 
        # the compiler can do optimizations
        start_kv = tl.multiple_of(start_kv, BLOCK_SIZE_KV)
        
        # --compute qk--
        K_block = tl.load(K_block_ptr)
        QK_block = tl.dot(Q_block, K_block)
        
        # --update accumulator--
        if STAGE == 2:
            mask = offs_q[:, None] >= (start_kv + offs_kv[None, :])
            QK_block = QK_block * softmax_scale + tl.where(mask, 0, -1.0e6)
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1)) # 这里的1是维度
            QK_block -= m_ij[:, None]
        else:
            # Compute the maximum value of qk or keep the old max value
            m_ij = tl.maximum(m_i, tl.max(QK_block, 1) * softmax_scale)
            QK_block = QK_block * softmax_scale - m_ij[:, None]
        
        # Compute the expotional of each dot product, so now we are computing exp(qk_ij - m_i)
        P_block = tl.math.exp(QK_block)
        
        # Compute the sum by rows of the attention score
        l_ij = tl.sum(P_block, 1)
        
        # This is the correction factor for the online softmax
        alpha = tl.math.exp(m_i - m_ij)
        
        # Apply the correction factor to the running sum
        l_i = l_i * alpha + l_ij
        
        V_block = tl.load(V_block_ptr)
        
        P_block = P_block.to(tl.float16)
        
        # This computes: O_new = P x V + O_old * alpha
        O_block = O_block * alpha[:, None] # 不用左乘 diag 了，直接广播乘
        O_block = tl.dot(P_block, V_block, O_block) # O_block += P_block @ V_block
        
        m_i = m_ij
        
        # Move to the next block of K and V
        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_SIZE_KV, 0)) # V: [SEQ_LEN, HEAD_DIM] 
        K_block_ptr = tl.advance(K_block_ptr, (0, BLOCK_SIZE_KV)) # K: [HEAD_DIM, SEQ_LEN] 
        
    return O_block, l_i, m_i
    
    

@triton.autotune(
    [
        triton.Config(
            {"BLOCK_SIZE_Q": BLOCK_SIZE_Q, "BLOCK_SIZE_KV": BLOCK_SIZE_KV},
            num_stages=num_stages,
            num_warps=num_warps,
        )
        for BLOCK_SIZE_Q in [64,128]
        for BLOCK_SIZE_KV in [32,64]
        for num_stages in [3,4,7]
        for num_warps in [2,4]
    ],
    key=["SEQ_LEN", "HEAD_DIM"],
)
@triton.jit
def _attn_fwd(
    Q, # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    K, # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    V, # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    softmax_scale, 
    M, # BATCH_SIZE, NUM_HEADS, SEQ_LEN
    O, # BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM
    stride_Q_batch, stride_Q_head, stride_Q_seq, stride_Q_dim,
    stride_K_batch, stride_K_head, stride_K_seq, stride_K_dim,
    stride_V_batch, stride_V_head, stride_V_seq, stride_V_dim,
    stride_O_batch, stride_O_head, stride_O_seq, stride_O_dim,
    BATCH_SIZE,
    NUM_HEADS: tl.constexpr,
    SEQ_LEN: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_SIZE_Q: tl.constexpr,
    BLOCK_SIZE_KV: tl.constexpr,
    STAGE: tl.constexpr,
):
    tl.static_assert(BLOCK_SIZE_KV <= HEAD_DIM, "BLOCK_SIZE_KV must be less than or equal to HEAD_DIM")
    
    # This indicate which block in the sequence length to process
    block_index_q = tl.program_id(0)
    # This indicates which head and batch to process. Each programme will process one head of one batch.
    index_batch_head = tl.program_id(1)
    # This indicates which batch this program is associated with (each batch has NUM_HEADS heads)
    index_batch = index_batch_head // NUM_HEADS
    # This indicates the position of the head in the batch
    index_head = index_batch_head % NUM_HEADS
    
    # This allows to get the (SEQ_LEN, HEAD_DIM) block in the Q,K,V by selecting indexing it by batch and head.
    qkv_offset = (
        index_batch.to(tl.int64) * stride_Q_batch +
        index_head.to(tl.int64) * stride_Q_head 
    )
    
    Q_block_ptr = tl.make_block_ptr( # Q[index_batch, index_head, block_index_q * BLOCK_SIZE_Q:, :]
        # 当前 Program 正在处理的那个 SEQ_LEN x HEAD_DIM 块的 第一个元素的物理地址。
        base = Q + qkv_offset, # create a pointer the first element of the block(tensor) Q[index_batch, index_head, :, :]
        # 从这个base地址开始，数据的逻辑形状是 SEQ_LEN x HEAD_DIM。程序正在处理的整个数据区域的边界。
        shape = (SEQ_LEN, HEAD_DIM), # shape of the block
        # 内部步长
        strides = (stride_Q_seq, stride_Q_dim), # strides of the block(tensor) Q[index_batch, index_head, :, :]
        # 当前programme 真正要从 SEQ_LEN x HEAD_DIM 块的 哪个位置开始读取数据。这个block，也就是把整个seq分成多个block来计算的block
        offsets = (block_index_q * BLOCK_SIZE_Q, 0), 
        # 每次访问要加载一个 BLOCK_SIZE_Q x HEAD_DIM 的块。这个块是 shape 的一部分。并行访问单元。
        block_shape = (BLOCK_SIZE_Q, HEAD_DIM),
        # row-major order: 先按行，再按列
        order = (1, 0),
    )
    
    V_block_ptr = tl.make_block_ptr( # V[index_batch, index_head, :, :]
        base = V + qkv_offset,
        shape = (SEQ_LEN, HEAD_DIM),
        strides = (stride_V_seq, stride_V_dim),
        offsets = (0, 0),
        block_shape = (BLOCK_SIZE_KV, HEAD_DIM),
        order = (1, 0),
    )
    
    K_block_ptr = tl.make_block_ptr( # K[index_batch, index_head, :, :]
        base = K + qkv_offset,
        shape = (HEAD_DIM, SEQ_LEN),
        strides = (
            stride_K_dim,
            stride_K_seq,
        ), # we transpose the matrix
        offsets = (0, 0),
        block_shape = (HEAD_DIM, BLOCK_SIZE_KV),
        order = (0, 1), # 第0为变得最快
    )
    
    O_block_ptr = tl.make_block_ptr( # O[index_batch, index_head, block_index_q * BLOCK_SIZE_Q:, :]
        base = O + qkv_offset,
        shape = (SEQ_LEN, HEAD_DIM),
        strides = (stride_O_seq, stride_O_dim),
        offsets = (block_index_q * BLOCK_SIZE_Q, 0),
        block_shape = (BLOCK_SIZE_Q, HEAD_DIM),
        order = (1, 0),
    )
    
    #########################################################
    ## 在内层循环，才对K和V进行分块，这里只需要对Q有offset        ##
    #########################################################
    
    # offs_q: the offsets for the tokens in the Q to process
    # 这个长度为 BLOCK_SIZE_Q 的一维张量，代表当前Q块对应的整个Q矩阵的行索引
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q)
    
    # offs_kv: the offsets for the tokens in the K and V to process
    offs_kv = tl.arange(0, BLOCK_SIZE_KV) # 需要全部iterate一遍，从头开始
    
    # m_i: the running maximum, We have one for each query.
    m_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) - float("inf")
    
    # l_i: the running sum, We have one for each query.
    l_i = tl.zeros([BLOCK_SIZE_Q], dtype=tl.float32) + 1.0
    
    # O_block: the running output, We have one for each query.
    O_block = tl.zeros([BLOCK_SIZE_Q, HEAD_DIM], dtype=tl.float32)
    
    # Load the Q block from the pointer, it will stay in SRAM throughput
    # Q_block_ptr 已经包含了所有信息
    Q_block = tl.load(Q_block_ptr)
    
    # 进入内层循环，online softmax，不断更新m，L和O
    
    # Stage: 3 if causal else 1
    
    if STAGE == 1 or STAGE == 3:
        # This step runs for non-causal or for the blocks to the left of the diagonal in the causal attention.
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i, 
            Q_block, # pass in Q_block instead of Q_block_ptr, to reduce the number of loads
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            4 - STAGE,
            offs_q,
            offs_kv,
            SEQ_LEN,
        )
    
    if STAGE == 3:
        # This step runs for the blocks to the right of the diagonal in the causal attention.
        O_block, l_i, m_i = _attn_fwd_inner(
            O_block,
            l_i,
            m_i, 
            Q_block,
            K_block_ptr,
            V_block_ptr,
            block_index_q,
            softmax_scale,
            BLOCK_SIZE_Q,
            BLOCK_SIZE_KV,
            2,
            offs_q,
            offs_kv,
            SEQ_LEN,
        )
    
    # Already computed the O and L
    # 为反向传播作准备
    # 计算 logsumexp，LSE = log(sum(e^{zj} for j))，也就是分母。
    # 对于 pi = softmax(zi), 如果i=k, \partial p \/ \partial z_i = p_i (1 - p_i)
    # 如果i!=k, \partial p \/ \partial z_i = -p_i \times p_k
    m_i += tl.math.log(l_i) # This is needed to compute the logsumexp for the backward pass
    # 还需要计算p_i，因此，防止第二次计算值和第一次计算差别过大，需要保留logsumexp的值。第二次直接用。
    # softmax(xi) = exp(xi - mi - log(li)) = exp(xi - mi) / li
    # 所以第二次可以很快恢复 softmax(xi) 的值。
    
    O_block = O_block / l_i[:, None] # 除以分母。
    
    # 把M与Qi所有行一一对应的M的物理地址
    m_ptrs = M + index_batch_head * SEQ_LEN + offs_q
    tl.store(m_ptrs, m_i)
    tl.store(O_block_ptr, O_block.to(O.type.element_ty))
    
    

@triton.jit
def _attn_bwd_preprocess(
    O,
    dO,
    D, # [BATCH_SIZE, NUM_HEADS, SEQ_LEN]
    SEQ_LEN,
    BLOCK_SIZE_Q: tl.constexpr,
    HEAD_DIM: tl.constexpr,
):
    # with pointer to O, dO. We need to compute D and store it in D. D is the shape of P, one row for each.
    block_index_q = tl.program_id(0) # which block of queries we are processing
    offs_q = block_index_q * BLOCK_SIZE_Q + tl.arange(0, BLOCK_SIZE_Q) # 要处理的一个block的Q的行索引
    index_batch_head = tl.program_id(1) # which head of which batch we are processing
    offs_dim = tl.arange(0, HEAD_DIM) # 要处理的一个block的Q的列索引
    
    # Load a single block of BLOCK_SIZE_Q rows of O. Remember that O is the same shape as Q. []
    O_block = tl.load( # the whole O: [BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM]
        O # 起始地址
        + index_batch_head * SEQ_LEN * HEAD_DIM # 当前的 index_batch_head 对应的 batch 的 起始地址
        + offs_q[:, None] * HEAD_DIM # 将行索引数组 乘以列的步长
        + offs_dim[None, :], # 将列索引数组 乘以行内的列偏移量。维度变换是为了相加。
    ) # [BLOCK_SIZE_Q, HEAD_DIM]
    # 现在整个块都load到了SRAM上。
    
    # Load a single block of BLOCK_SIZE_Q rows of dO.
    dO_block = tl.load(
        dO
        + index_batch_head * SEQ_LEN * HEAD_DIM
        + offs_q[:, None] * HEAD_DIM
        + offs_dim[None, :],
    ).to(tl.float32) # [BLOCK_SIZE_Q, HEAD_DIM]
    
    # Compute the D block
    D_block = tl.sum(dO_block * O_block, axis=1)  # [BLOCK_SIZE_Q]. ALL the D_i, one for each.
    # D 和 M 同形状, [BATCH_SIZE, NUM_HEADS, SEQ_LEN]
    # Store the D block
    D_block_ptrs = D + index_batch_head * SEQ_LEN + offs_q
    tl.store(D_block_ptrs, D_block)
    

@triton.jit
def _attn_bwd_dk_dv_inner(
    Q,
    K,
    V,
    softmax_scale,
    dO,
    dQ,
    dK,
    dV,
    M,
    D,
    stride_batch, stride_head, stride_seq, stride_dim,
    NUM_HEADS,
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
):
    index_batch_head = tl.program_id(2) # 对应grid的形状，[SEQ_LEN // BLOCK_SIZE_MACRO, 1, BATCH_SIZE * NUM_HEADS]的第二维。
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS
    offset_batch_head = (stride_batch * index_batch + stride_head * index_head).to(tl.int64)
    # This is the offset that allows us to select the right sequence given the batch and head.
    offset_bach_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64) # for M and D
    
    # Make sure the pointers are in the right place w.r.t batch and head.
    # The reason we don't access the blocks through make_block_ptr is because we need to use the range of offsets to apply the masking
    Q += offset_batch_head
    K += offset_batch_head # [B, NUM_HEADS, SEQ_LEN, HEAD_DIM]
    V += offset_batch_head
    dO += offset_batch_head
    dQ += offset_batch_head
    dK += offset_batch_head
    dV += offset_batch_head
    
    # Make sure the pointers are in the right place w.r.t batch, head and sequence.
    M += offset_bach_head_seq
    D += offset_bach_head_seq
    
    # load scales
    offs_dim = tl.arange(0, HEAD_DIM)
    
    index_block_kv = tl.program_id(0)
    start_kv = index_block_kv * BLOCK_KV
    
    offs_kv = start_kv + tl.arange(0, BLOCK_KV)
    
    dV_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)
    dK_block = tl.zeros([BLOCK_KV, HEAD_DIM], dtype=tl.float32)
    
    # load K and V: They stay in SRAM throughout the inner loop
    K_block = tl.load(K + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim) # [BLOCK_KV1, HEAD_DIM]
    V_block = tl.load(V + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim) # [BLOCK_KV1, HEAD_DIM]
    
    offs_q = tl.arange(0, BLOCK_Q) # Block size micro
    
    # We access the Q as a transposed array, so that's why we treat offs_q as a column vector and offs_dim as a row vector
    # This is same as: q_ptrs = Q + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    # qT_ptrs = tl.trans(q_ptrs)
    # We point to the first BLOCK_Q rows of Q for both the qT and dO pointers, inside the for loop we will move forward by BLOCK_Q rows at each iteration.
    qT_ptrs = Q + offs_q[None, :] * stride_seq + offs_dim[:, None] * stride_dim
    dO_ptrs = dO + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    
    # Iterates over the sequence dimension of the query
    curr_q = 0
    num_steps = SEQ_LEN // BLOCK_Q
    for blk_idx in range(num_steps):
        # Load
        qT_block = tl.load(qT_ptrs)
        # Load the LSE for the queries in the current block
        offs_q = curr_q + tl.arange(0, BLOCK_Q)
        m = tl.load(M + offs_q)
        
        # This gives us (QKT)T = KQT = PT
        QK_T_block = softmax_scale * tl.dot(K_block, qT_block) # recompute instead of store in SRAM. Calculate is faster than reading from HBM.
        # We apply the softmax by using the LSE trick. 上面fwd的注释里有。恢复P
        P_T_block = tl.math.exp(QK_T_block - m[None, :])
        
        if STAGE == 3:
            # Autoregressive masking
            # mask is True for all values that DO NOT need to be masked.
            mask_block = (
                offs_q[None, :] >= offs_kv[:, None]
            ) # [BLOCK_KV1, BLOCK_Q]
            # Replace all the masked with 0
            P_T_block = tl.where(mask_block, P_T_block, 0.0) # not use -inf because softmax is already applied
        
        dO_block = tl.load(dO_ptrs)
        # Formula: dV_new = dV_old + P^T x dO
        dV_block += tl.dot(P_T_block.to(tl.float16), dO_block)
        
        # Di = rowsum(O * dO) where * is element-wise multiplication
        Di = tl.load(D + offs_q)
        
        # dP = dO x V^T. so dP^T = V x dO^T
        dpT_block = tl.dot(V_block, tl.trans(dO_block)).to(tl.float32)
        
        # dS = P * (dP - Di), so dS^T = P^T * (dP^T - Di^T)
        dS_T_block = P_T_block * (dpT_block - Di[None, :])
        dS_T_block = dS_T_block.to(tl.float16)
        
        # Formula: dK_new = dK_old + dS^T x Q
        dK_block += softmax_scale * tl.dot(dS_T_block, tl.trans(qT_block))
        # Increment the pointers
        curr_q += BLOCK_Q
        qT_ptrs += BLOCK_Q * stride_seq
        dO_ptrs += BLOCK_Q * stride_seq
    
    # Write back dK and dV
    dV_block_ptrs = dV + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim # kv 的block 的所有地址
    tl.store(dV_block_ptrs, dV_block)
    dK_block_ptrs = dK + offs_kv[:, None] * stride_seq + offs_dim[None, :] * stride_dim
    tl.store(dK_block_ptrs, dK_block)
    

@triton.jit
def _attn_bwd_dq(
    Q,
    K,
    V,
    softmax_scale,
    dO,
    dQ,
    dK,
    dV,
    M,
    D,
    stride_batch, stride_head, stride_seq, stride_dim,
    NUM_HEADS,
    SEQ_LEN,
    BLOCK_Q: tl.constexpr,
    BLOCK_KV: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    STAGE: tl.constexpr,
):
    index_batch_head = tl.program_id(2) # 对应grid的形状，[SEQ_LEN // BLOCK_SIZE_MACRO, 1, BATCH_SIZE * NUM_HEADS]的第二维。
    index_batch = index_batch_head // NUM_HEADS
    index_head = index_batch_head % NUM_HEADS
    offset_batch_head = (stride_batch * index_batch + stride_head * index_head).to(tl.int64)
    # This is the offset that allows us to select the right sequence given the batch and head.
    offset_bach_head_seq = (index_batch_head * SEQ_LEN).to(tl.int64) # for M and D
    
    # Make sure the pointers are in the right place w.r.t batch and head.
    # The reason we don't access the blocks through make_block_ptr is because we need to use the range of offsets to apply the masking
    Q += offset_batch_head
    K += offset_batch_head # [B, NUM_HEADS, SEQ_LEN, HEAD_DIM]
    V += offset_batch_head
    dO += offset_batch_head
    dQ += offset_batch_head
    dK += offset_batch_head
    dV += offset_batch_head
    
    # Make sure the pointers are in the right place w.r.t batch, head and sequence.
    M += offset_bach_head_seq
    D += offset_bach_head_seq
    
    # load scales
    offs_dim = tl.arange(0, HEAD_DIM)
    
    index_block_q = tl.program_id(0)
    
    start_q = index_block_q * BLOCK_Q
    offs_q = start_q + tl.arange(0, BLOCK_Q)
    
    Q_block = tl.load(Q + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim) # [BLOCK_Q, HEAD_DIM]
    dQ_block = tl.zeros([BLOCK_Q, HEAD_DIM], dtype=tl.float32)
    dO_block = tl.load(dO + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim) # [BLOCK_Q, HEAD_DIM]
    
    M_block = tl.load(M + offs_q)
    M_block = M_block[:, None]
    
    offs_kv = tl.arange(0, BLOCK_KV)
        
    # We access K and V as a transposed array
    kT_ptrs = K + offs_kv[None, :] * stride_seq + offs_dim[:, None] * stride_dim
    vT_ptrs = V + offs_kv[None, :] * stride_seq + offs_dim[:, None] * stride_dim
    
    Di = tl.load(D + offs_q)
    # Iterates over the sequence dimension of kv
    curr_kv = 0
    num_steps = SEQ_LEN // BLOCK_KV
    
    for blk_idx in range(num_steps):
        K_T_block = tl.load(kT_ptrs)
        V_T_block = tl.load(vT_ptrs)
        
        QK_block = softmax_scale * tl.dot(Q_block, K_T_block)
        P_block = tl.math.exp(QK_block - M_block) # using LSE
        
        if STAGE == 3:
            # Autoregressive masking
            offs_kv = curr_kv + tl.arange(0, BLOCK_KV)
            mask_block = offs_q[:, None] >= offs_kv[None, :]
            P_block = tl.where(mask_block, P_block, 0.0)
            
        # Compute dP and dS
        dP_block = tl.dot(dO_block, V_T_block).to(tl.float32)
        dS_block = P_block * (dP_block - Di[:, None])
        dS_block = dS_block.to(tl.float16)
        
        # Formula: dQ_new = dQ_old + dS x K
        dQ_block += softmax_scale * tl.dot(dS_block, tl.trans(K_T_block))
        
        # Increment the pointers
        curr_kv += BLOCK_KV
        kT_ptrs += BLOCK_KV * stride_seq
        vT_ptrs += BLOCK_KV * stride_seq
        
        # Write back dQ
        dQ_block_ptrs = dQ + offs_q[:, None] * stride_seq + offs_dim[None, :] * stride_dim
        tl.store(dQ_block_ptrs, dQ_block)
        
        
    
    

class TritonAttention(torch.autograd.Function):
    
    @staticmethod
    def forward(ctx, Q, K, V, causal, softmax_scale):
        # ctx: something that we can use to store information for the backward pass
        HEAD_DIM_Q, HEAD_DIM_K, HEAD_DIM_V = Q.shape[-1], K.shape[-1], V.shape[-1]
        
        BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM = Q.shape
        
        assert HEAD_DIM_Q == HEAD_DIM_K == HEAD_DIM_V
        
        O = torch.empty_like(Q)
        
        stage = 3 if causal else 1
        
        # Launch grid
        # How many parallel processes we need to be lauched by triton
        grid = lambda args: (
            # ceil(SEQ_LEN / BLOCK_SIZE_Q) = How many groups of queries we are processing
            triton.cdiv(SEQ_LEN, args["BLOCK_SIZE_Q"]), # Which group of queries we are processing
            BATCH_SIZE * NUM_HEADS, # Which head of which batch we are processing
            1,
            # 定义了一个3D网格，每个 Program 会拿到自己的三维 ID (pid_0, pid_1, pid_2)
        )
        
        # M is the logsumexp for the backward pass, one for each query
        M = torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN), dtype=torch.float32, device=Q.device
        )
        
        _attn_fwd[grid]( # 在GPU上启动一个名为_attn_fwd的Triton Kernel。[grid]告诉triton运行时系统启动多少个并行的program。
            # 参数包含三种
            # 1. 数据指针：tensor的起始地址指针
            Q=Q,
            K=K,
            V=V,
            softmax_scale=softmax_scale,
            M=M,
            O=O,
            # 2. 步长信息：tensor的元素在内存中的偏移量
            stride_Q_batch=Q.stride(0),
            stride_Q_head=Q.stride(1),
            stride_Q_seq=Q.stride(2),
            stride_Q_dim=Q.stride(3),
            stride_K_batch=K.stride(0),
            stride_K_head=K.stride(1),
            stride_K_seq=K.stride(2),
            stride_K_dim=K.stride(3),
            stride_V_batch=V.stride(0),
            stride_V_head=V.stride(1),
            stride_V_seq=V.stride(2),
            stride_V_dim=V.stride(3),
            stride_O_batch=O.stride(0),
            stride_O_head=O.stride(1),
            stride_O_seq=O.stride(2),
            stride_O_dim=O.stride(3),
            # 3. 元数据和配置
            BATCH_SIZE=Q.shape[0],
            NUM_HEADS=Q.shape[1],
            SEQ_LEN=Q.shape[2],
            HEAD_DIM=HEAD_DIM_K,
            STAGE=stage,
        )
        
        ctx.save_for_backward(Q, K, V, O, M)
        ctx.grid = grid
        ctx.softmax_scale = softmax_scale
        ctx.HEAD_DIM = HEAD_DIM_K
        ctx.causal = causal
        return O
    
    @staticmethod
    def backward(ctx, dO):
        Q, K, V, O, M = ctx.saved_tensors
        
        assert dO.is_contiguous()
        assert Q.stride() == K.stride() == V.stride() == O.stride() == dO.stride()
        dQ = torch.empty_like(Q)
        dK = torch.empty_like(K)
        dV = torch.empty_like(V)
        
        BATCH_SIZE, NUM_HEADS, SEQ_LEN = Q.shape[:3]
        NUM_WARPS, NUM_STAGES = 4, 3  # warps:how many threads we want to launch in each grid, stages: software pipelining
        BLOCK_SIZE_MICRO, BLOCK_SIZE_MACRO = 32, 128 # the one we fix is the macro one, the one we iterate is the micro one
        
        preprocess_grid = (SEQ_LEN // BLOCK_SIZE_MACRO, BATCH_SIZE * NUM_HEADS)
        D = torch.empty_like(M) # [batch size, num heads, seq len]
        
        # Compute all the elements Di
        _attn_bwd_preprocess[preprocess_grid](
            O=O,
            dO=dO,
            D=D,
            SEQ_LEN=SEQ_LEN,
            BLOCK_SIZE_Q=BLOCK_SIZE_MACRO,
            HEAD_DIM=ctx.HEAD_DIM,
        )
        
        # 2 loops
        grid = (SEQ_LEN // BLOCK_SIZE_MACRO, 1, BATCH_SIZE * NUM_HEADS) # 第一维决定多少个programms 并行。grid约定为3D
        
        stage = 3 if ctx.causal else 1
        
        # Fix KV, iterate Q, compute dK and dV
        _attn_bwd_dk_dv_inner[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=ctx.softmax_scale,
            dO=dO,
            dQ=dQ,
            dK=dK,
            dV=dV,
            M=M,
            D=D,
            stride_batch=Q.stride(0),
            stride_head=Q.stride(1),
            stride_seq=Q.stride(2),
            stride_dim=Q.stride(3),
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            BLOCK_Q=BLOCK_SIZE_MICRO,
            BLOCK_KV=BLOCK_SIZE_MACRO,
            HEAD_DIM=ctx.HEAD_DIM,
            STAGE=stage,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )
        
        # Fix Q, iterate KV, compute dQ
        _attn_bwd_dq[grid](
            Q=Q,
            K=K,
            V=V,
            softmax_scale=ctx.softmax_scale,
            dO=dO,
            dQ=dQ,
            dK=dK,
            dV=dV,
            M=M,
            D=D,
            stride_batch=Q.stride(0),
            stride_head=Q.stride(1),
            stride_seq=Q.stride(2),
            stride_dim=Q.stride(3),
            NUM_HEADS=NUM_HEADS,
            SEQ_LEN=SEQ_LEN,
            BLOCK_Q=BLOCK_SIZE_MACRO,
            BLOCK_KV=BLOCK_SIZE_MICRO,
            HEAD_DIM=ctx.HEAD_DIM,
            STAGE=stage,
            num_warps=NUM_WARPS,
            num_stages=NUM_STAGES,
        )
        return dQ, dK, dV, None, None
        

def test_op(BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM, causal, dtype=torch.float16):
    Q = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std = 0.5)
        .requires_grad_() # require
    )
    K = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std = 0.5)
        .requires_grad_()
    )
    V = (
        torch.empty(
            (BATCH_SIZE, NUM_HEADS, SEQ_LEN, HEAD_DIM), dtype=dtype, device="cuda"
        )
        .normal_(mean=0.0, std = 0.5)
        .requires_grad_()
    )
    
    softmax_scale = 1 / HEAD_DIM ** 0.5
    dO = torch.randn_like(Q) # Needed for backward pass。因为没有损失函数，所以假设dO = \partial L / \partial O
    
    # reference implementation
    MASK = torch.tril(torch.ones((SEQ_LEN, SEQ_LEN), dtype=dtype, device="cuda")) # 下三角置为1
    # [b, num_heads, seq_len, seq_len]
    P = torch.matmul(Q, K.transpose(-2, -1)) * softmax_scale
    if causal:
        P[:, :, MASK==0] = float("-inf")
    P = torch.softmax(P.float(), dim=-1).half()
    ref_O = torch.matmul(P, V)
    ref_O.backward(dO)
    ref_dV, V.grad = V.grad.clone(), None # 保存梯度并清理
    ref_dK, K.grad = K.grad.clone(), None
    ref_dQ, Q.grad = Q.grad.clone(), None
    
    # triton implementation
    tri_out = TritonAttention.apply(Q, K, V, causal, softmax_scale).half()
    tri_out.backward(dO)
    tri_dv, V.grad = V.grad.clone(), None
    tri_dk, K.grad = K.grad.clone(), None
    tri_dq, Q.grad = Q.grad.clone(), None
    
    rtol = 0.0
    atol = 1e-2
    assert torch.allclose(ref_O, tri_out, rtol=rtol, atol=atol)
    assert torch.allclose(ref_dV, tri_dv, rtol=rtol, atol=atol)
    assert torch.allclose(ref_dK, tri_dk, rtol=rtol, atol=atol)
    assert torch.allclose(ref_dQ, tri_dq, rtol=rtol, atol=atol)
    print("Test passed!")
    return True


if __name__ == "__main__":
    test_op(BATCH_SIZE=8, NUM_HEADS=16, SEQ_LEN=512, HEAD_DIM=32, causal=True)
    test_op(BATCH_SIZE=8, NUM_HEADS=16, SEQ_LEN=512, HEAD_DIM=32, causal=False)
    print("Test passed!")