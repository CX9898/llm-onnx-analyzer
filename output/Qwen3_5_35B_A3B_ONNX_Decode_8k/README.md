# Qwen3.5-MoE Merged ONNX 说明：`decode`

这个目录保存的是 `Qwen3.5-35B-A3B` 一套面向 `8k decode` 的代表层 merged ONNX 子图，以及配套的统计与流程分析文件。

## 完整数据流（按这个顺序看 onnx）

下图覆盖本目录所有主链 onnx 文件，箭头方向就是 decode 单步 forward 时数据流的方向；列在右侧的 `*.onnx` 就是每一步对应的产物文件名。

```text
[Decode — 每步只前进 1 个新 token，历史信息全部走 KV-cache / recurrent state]

  input_ids:[1,1]i64
        │
        ▼  embedding_1.onnx
  inputs_embeds:[1,1,2048]bf16
        │
        │   ×40 层 decoder stack (layer_idx % 4 == 3 → full attention, 否则 linear attention)
        │   本目录只导出两类代表层:
        │
        │     layer 0 (linear attn + MoE):
        │       ─► layer_00_linear_attn_block.onnx
        │            states (in/out): conv_state:[1,8192,4]bf16,
        │                             recurrent_state:[1,32,128,128]float32
        │       ─► layer_00_moe_block_1.onnx
        │
        │     layer 3 (full attn + MoE):
        │       ─► layer_03_full_attn_block_decode_ctx8k.onnx
        │            cache (in/out): past_key/past_value:[1,2,8192,256]bf16
        │            attention_mask:[1,1,1,8193]bf16
        │       ─► layer_03_moe_block_1.onnx
        │
        ▼  norm_1.onnx
  hidden_states:[1,1,2048]bf16
        │
        ▼  lm_head_1.onnx
  logits:[1,1,248320]bf16
```

主链之外，本目录还导出一张 **custom op 展开参考子图**（不是独立调用步骤，只是 `layer_00_linear_attn_block.onnx` 顶层图里 `qwen_onnx::RecurrentGatedDeltaRule` 的内部展开，方便结构对照）：

- `layer_00_linear_attn_block_RecurrentGatedDeltaRule.onnx`

> 注：这两类代表层是 **并列样本**，不是 `layer_00_*` 跑完接到 `layer_03_*` 的顺序链路。真实的 40 层 decoder 是 `layer_idx=0,1,2(linear) → 3(full) → 4,5,6(linear) → 7(full) → ...` 这样不断重复，本目录只各取一种代表层导出。

## 先看结论

- `decode` 的意思是：每一步只处理 `1` 个新 token，但会携带历史状态继续递推。
- 本目录里最值得关注的是四张主图：
  - `layer_00_linear_attn_block.onnx`
  - `layer_00_moe_block_1.onnx`
  - `layer_03_full_attn_block_decode_ctx8k.onnx`
  - `layer_03_moe_block_1.onnx`
- `layer_00_linear_attn_block_RecurrentGatedDeltaRule.onnx` 不是主链 block，而是顶层 custom op 的参考展开图。
- `embedding_1.onnx`、`norm_1.onnx`、`lm_head_1.onnx` 主要是补齐完整首尾链路，不是这个目录最核心的差异点。

## 这套目录在整模型里代表什么

Qwen3.5-MoE 的 decoder stack 一共 `40` 层，每层都包含：

- attention token mixer
- 后续 MoE FFN

其中 token mixer 按“每 `3` 层 `linear attention`，再 `1` 层 `full attention`”的节奏重复。当前目录只导出两类代表层：

- `layer 0`：代表 `linear attention + MoE`
- `layer 3`：代表 `full attention + MoE`

也就是说，这个目录不是完整 40 层模型，而是“能代表两类层结构”的一组样本图。

## `decode` 在这里具体是什么意思

在这套导出里，`decode` 表示：

- 当前步只输入一个新 token
- 当前主链激活张量是 `seq_len=1`
- 历史信息通过状态张量保留下来，而不是把整段历史 hidden states 再输入一次

因此读这个目录时，最重要的直觉应该是：

- 这是一套“单 token 递推”的图
- 但它不是“无历史”的图，而是“带历史状态”的图

## 主链里哪些图最重要

对大多数读者，优先关注下面四张图就够了：

### `layer_00_linear_attn_block.onnx`

- 第 0 层 merged linear attention block
- 等价于 `input_norm -> DeltaNet(decode) -> residual add`
- 主干输入输出都是 `seq_len=1`

### `layer_00_moe_block_1.onnx`

- 第 0 层 merged MoE block
- 等价于 `post_norm -> moe_ffn -> residual add`

### `layer_03_full_attn_block_decode_ctx8k.onnx`

- 第 3 层 merged full attention block
- 等价于 `input_norm -> rotary -> self_attn -> residual add`
- 这是当前目录里最显式表达“8k 历史上下文”的主图

### `layer_03_moe_block_1.onnx`

- 第 3 层 merged MoE block
- 等价于 `post_norm -> moe_ffn -> residual add`

## 这些主图之间怎么理解

这里有两条**并列的代表层样本路径**，不是一条要顺序首尾相连执行的局部网络：

### `layer 0` 样本路径

`embedding_1 -> layer_00_linear_attn_block -> layer_00_moe_block_1`

### `layer 3` 样本路径

`layer_03_full_attn_block_decode_ctx8k -> layer_03_moe_block_1`

重点是：

- `layer_00_*` 和 `layer_03_*` 是并列代表样本
- 不是“当前目录里先跑完 `layer_00_*` 再去接 `layer_03_*`”

## 你真正需要关心的 `decode` 差异

在 `decode` 目录里，最重要的区别是：

- 主链 `hidden_states` 都是 `seq_len=1`
- 历史信息不再作为整段 hidden states 输入，而是通过状态量表达

但是两类 attention 的方式不同：

### full attention

`layer_03_full_attn_block_decode_ctx8k.onnx` 会显式把历史 KV cache 作为图输入：

- `past_key:[1, 2, 8192, 256]`
- `past_value:[1, 2, 8192, 256]`
- `attention_mask:[1, 1, 1, 8193]`

所以 `full attention` 的 `8k` 差异是直接体现在图接口里的。

### linear attention

`layer_00_linear_attn_block.onnx` 不会额外带一个“历史长度为 8192”的显式轴，而是依赖：

- `conv_state`
- `recurrent_state`

所以 `linear attention` 在 `decode` 下的 `8k` 差异，体现在**状态内容**，不体现在一个单独的历史长度维度上。

## 有一个最容易误解的点

`layer_00_linear_attn_block.onnx` 里会看到：

- `conv_state:[1, 8192, 4]`
- `recurrent_state:[1, 32, 128, 128]`

这里的 `conv_state` 中间那个 `8192`，**不是“8k 历史 token 长度”**，而是这个实现里的卷积通道维度。  
也就是说：

- `past_key / past_value` 里的 `8192` 才是显式历史序列长度
- `conv_state` 里的 `8192` 不是同一个含义

## custom op 参考子图是什么

### `layer_00_linear_attn_block_RecurrentGatedDeltaRule.onnx`

这个文件的意义是：

- 它不是主链上的独立 block
- 它是顶层 `layer_00_linear_attn_block.onnx` 中 `qwen_onnx::RecurrentGatedDeltaRule` 的参考展开图
- 主要用于：
  - 结构分析
  - custom op 对照
  - 流程表细看内部实现

所以如果你只是想理解“主链长什么样”，这张图不是第一优先级。

## 逐文件说明

下面按“主链图优先、辅助图后置”的顺序说明。

### `embedding_1.onnx`

- 作用：token embedding
- 输出：`hidden_states:[1, 1, 2048]`

### `layer_00_linear_attn_block.onnx`

- 作用：第 0 层 merged linear attention block
- 输入：
  - `hidden_states:bf16:[1, 1, 2048]`
  - `conv_state:bf16:[1, 8192, 4]`
  - `recurrent_state:float:[1, 32, 128, 128]`
  - `padding_mask:bool:[1, 1]`
- 输出：
  - `hidden_states:bf16:[1, 1, 2048]`
  - `new_conv_state:bf16:[1, 8192, 4]`
  - `new_recurrent_state:float:[1, 32, 128, 128]`
- 说明：
  - 已经融合 `input_norm`
  - 顶层图中包含一个 `qwen_onnx::RecurrentGatedDeltaRule`
  - 主输出回模型主 dtype
  - `new_recurrent_state` 保持 `FLOAT`，和源码 recurrence 语义一致

### `layer_00_moe_block_1.onnx`

- 作用：第 0 层 merged MoE block
- 输入：主干输入 `hidden_states`
- 输出：`hidden_states`

### `layer_03_full_attn_block_decode_ctx8k.onnx`

- 作用：第 3 层 merged full attention block
- 输入：
  - `hidden_states:bf16:[1, 1, 2048]`
  - `position_ids:int64:[1, 1]`
  - `attention_mask:bf16:[1, 1, 1, 8193]`
  - `past_key:bf16:[1, 2, 8192, 256]`
  - `past_value:bf16:[1, 2, 8192, 256]`
- 输出：
  - `hidden_states:bf16:[1, 1, 2048]`
  - `new_key:bf16:[1, 2, 8193, 256]`
  - `new_value:bf16:[1, 2, 8193, 256]`
- 说明：
  - 这是当前目录里最显式表达“8k 历史上下文”的图
  - `new_key / new_value` 是更新后的 KV cache，不是给下一层 MoE block 的主输入

### `layer_03_moe_block_1.onnx`

- 作用：第 3 层 merged MoE block
- 输入：主干输入 `hidden_states`
- 输出：`hidden_states`

### `norm_1.onnx`

- 作用：final norm
- 输入：`hidden_states:[1, 1, 2048]`
- 输出：`output:[1, 1, 2048]`

### `lm_head_1.onnx`

- 作用：lm head
- 输入：`hidden_states:[1, 1, 2048]`
- 输出：`logits:[1, 1, vocab_size]`

### `layer_00_linear_attn_block_RecurrentGatedDeltaRule.onnx`

- 作用：`RecurrentGatedDeltaRule` custom op 参考子图
- 输入：
  - `query/key/value/g/beta:bf16`
  - `recurrent_state:float`
- 输出：
  - `core_out:bf16`
  - `new_recurrent_state:float`
- 说明：
  - 对外接口遵循“主输出回主 dtype、state 保持 float”的源码语义

## 目录里的辅助文件

- `layer_00_linear_attn_block.json`
  - structured manifest
  - 用来描述顶层主图和 `RecurrentGatedDeltaRule` 参考图之间的关系
- `onnx_stats.json`
  - 每个导出文件的基础统计
- `onnx_flow_stats_multi.xlsx`
  - 批量流程/算子统计报表
