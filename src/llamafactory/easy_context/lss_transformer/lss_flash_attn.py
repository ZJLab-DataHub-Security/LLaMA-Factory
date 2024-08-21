import torch 
import torch.distributed as dist
from flash_attn.flash_attn_interface import _flash_attn_forward, _flash_attn_backward


def lss_flash_attn_forward(
        process_group,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        sofmax_scale,
        dropout_p = 0,
        causal = True,
        window_size =  (-1, -1),
        alibi_slopes = None,
        deterministic = False,
): 
    '''
    k: k_proj
    v: v_proj 
    1. 将L * self-attention的入口处layer Norm, pos emb, token emb的lookup table分散到不同的rank
    上 
    2. Fused communication,根据sp分散出x_i 
        2.1 计算出Q_i. 
        2.2 all-gather出x,计算出完整的KV (1次communication)(可overlap)

    与flash_attn 返回一致 

    假设: 先不考虑1.,仅考虑lss_flash_attn的self-attention部分 

    '''
    # 实现allgather逻辑
    # 1. 获取rank
    rank = dist.get_rank(process_group)
    world_size = dist.get_world_size(process_group)
    # 2. 获得x_i的形状，x_i.shape = q.shaoe
    x_i_shape = q.shape
    # 3. 根据x_i的形状,获得x的形状 x_i.shape = [b, a, s_i, d]
    # x_shape = x_i.shape * world_size
    x_shape = [x_shape[0], x_shape[1], x_shape[2]*world_size, x_shape[3]]
    # 4. 创建空的x
    x = torch.zeros(x_shape)
    # 5. 使用allgather通信来完成x的收集
    

    x = dist.all_gather()

    # 使用flash_attn
    rank_out ,_, _, _, _, _, _, _ = _flash_attn_forward(
        q,
        k,
        v,
        dropout_p,
        sofmax_scale,
        causal,
        window_size,
        alibi_slopes,
        return_softmax=True and dropout_p > 0,
    )

    rank_out = rank_out.to(q.dtype)

    pass

    

def lss_flash_attn_backward(
        process_group,
        dout,
        q,
        k,
        v,
        out,
        softmax_lse,
        softmax_scale,
        dropout_p = 0,
        causal=True,
        window_size = (-1, -1),
        alibi_slops = None,
        deterministic = False,
):
    #todo: 填充backward实现
    pass

class LssFlashAttnFunc(torch.autograd.Function):
    @staticmethod
    def forward(ctx,
                q,
                k,
                v,
                dropout_p,
                softmax_scale,
                causal,
                window_size,
                alibi_slopes,
                deterministic,
                return_softmax,
                group,
    ):
        if softmax_scale is None:
            softmax_scale = q.shape[-1] ** (-0.5)

        assert alibi_slopes is None
        k = k.contiguous()
        v = v.contiguous()
        out, softmax_lse = lss_flash_attn_forward(
            group,
            q,
            k,
            v,
            softmax_scale=softmax_scale,
            dropout_p=dropout_p,
            causal=causal,
            window_size=window_size,
            alibi_slopes=alibi_slopes,
            deterministic=False,
        )
        # this should be out_padded
        ctx.save_for_backward(q, k, v, out, softmax_lse)
        ctx.dropout_p = dropout_p
        ctx.softmax_scale = softmax_scale
        ctx.causal = causal
        ctx.window_size = window_size
        ctx.alibi_slopes = alibi_slopes
        ctx.deterministic = deterministic
        ctx.group = group
        return out if not return_softmax else (out, softmax_lse, None)
    
    @staticmethod
    def backward(ctx, dout, *args):
        q, k, v, out, softmax_lse = ctx.saved_tensors
        dq, dk, dv = lss_flash_attn_backward(
            ctx.group,
            dout,
            q,
            k,
            v,
            out,
            softmax_lse,
            softmax_scale=ctx.softmax_scale,
            dropout_p=ctx.dropout_p,
            causal=ctx.causal,
            window_size=ctx.window_size,
            alibi_slopes=ctx.alibi_slopes,
            deterministic=ctx.deterministic,
        )
        return dq, dk, dv, None, None, None, None, None, None, None, None

def lss_flash_attn_qkvpacked_func(
    qkv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    return LssFlashAttnFunc.apply(
        qkv[:, :, 0],
        qkv[:, :, 1],
        qkv[:, :, 2],
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )

def lss_flash_attn_kvpacked_func(
    q,
    kv,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    return LssFlashAttnFunc.apply(
        q,
        kv[:, :, 0],
        kv[:, :, 1],
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )

def lss_flash_attn_func(
    q,
    k,
    v,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=(-1, -1),
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    group=None,
):
    return LssFlashAttnFunc.apply(
        q,
        k,
        v,
        dropout_p,
        softmax_scale,
        causal,
        window_size,
        alibi_slopes,
        deterministic,
        return_attn_probs,
        group,
    )