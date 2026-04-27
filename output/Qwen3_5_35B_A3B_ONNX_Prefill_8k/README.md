# Qwen3.5-MoE Merged ONNX 说明：`prefill`

这个目录保存的是 `Qwen3.5-35B-A3B` 一套面向 `8k prefill` 的代表层 merged ONNX 子图，以及配套的统计与流程分析文件。

## 完整数据流（按这个顺序看 onnx）

下图覆盖本目录所有主链 onnx 文件，箭头方向就是 forward 时数据流的方向；列在右侧的 `*.onnx` 就是每一步对应的产物文件名。

```text
[Prefill — 整段 8k 上下文一次性灌入]

  input_ids:[1,8192]i64
        │
        ▼  embedding_8k.onnx
  inputs_embeds:[1,8192,2048]bf16
        │
        │   ×40 层 decoder stack (layer_idx % 4 == 3 → full attention, 否则 linear attention)
        │   本目录只导出两类代表层:
        │
        │     layer 0 (linear attn + MoE):
        │       ─► layer_00_linear_attn_block_8k.onnx
        │       ─► layer_00_moe_block_8k.onnx
        │
        │     layer 3 (full attn + MoE):
        │       ─► layer_03_full_attn_block_8k.onnx
        │       ─► layer_03_moe_block_8k.onnx
        │
        ▼  norm_8k.onnx
  hidden_states:[1,8192,2048]bf16
        │
        ▼  lm_head_8k.onnx
  logits:[1,8192,248320]bf16
```

主链之外，本目录还导出两张 **custom op 展开参考子图**（不是独立调用步骤，只是 `layer_00_linear_attn_block_8k.onnx` 顶层图里 `qwen_onnx::ChunkGatedDeltaRule` / `DeltaNetChunkStep` 的内部展开，方便结构对照）：

- `layer_00_linear_attn_block_ChunkGatedDeltaRule_chunk64_8k.onnx`
- `layer_00_linear_attn_block_ChunkGatedDeltaRule_DeltaNetChunkStep_chunk64.onnx`

> 注：这两类代表层是 **并列样本**，不是 `layer_00_*` 跑完接到 `layer_03_*` 的顺序链路。真实的 40 层 decoder 是 `layer_idx=0,1,2(linear) → 3(full) → 4,5,6(linear) → 7(full) → ...` 这样不断重复，本目录只各取一种代表层导出。

## 先看结论

- `prefill` 的意思是：一次输入整段上下文，这里固定是 `seq_len=8192`。
- 本目录里最值得关注的是四张主图：
  - `layer_00_linear_attn_block_8k.onnx`
  - `layer_00_moe_block_8k.onnx`
  - `layer_03_full_attn_block_8k.onnx`
  - `layer_03_moe_block_8k.onnx`
- `layer_00_linear_attn_block_ChunkGatedDeltaRule_chunk64_8k.onnx` 不是主链 block，而是顶层 custom op 的参考展开图。
- `layer_00_linear_attn_block_ChunkGatedDeltaRule_DeltaNetChunkStep_chunk64.onnx` 是更细一层的参考子图，用来对照 `ChunkStep` 内部递推。
- `DeltaNetTriangularSolve` 独立参考子图在这套目录里默认不导出。
- `embedding_8k.onnx`、`norm_8k.onnx`、`lm_head_8k.onnx` 主要是补齐完整首尾链路，不是这个目录最核心的差异点。

## 这套目录在整模型里代表什么

Qwen3.5-MoE 的 decoder stack 一共 `40` 层，每层都包含：

- attention token mixer
- 后续 MoE FFN

其中 token mixer 按“每 `3` 层 `linear attention`，再 `1` 层 `full attention`”的节奏重复。当前目录只导出两类代表层：

- `layer 0`：代表 `linear attention + MoE`
- `layer 3`：代表 `full attention + MoE`

也就是说，这个目录不是完整 40 层模型，而是“能代表两类层结构”的一组样本图。

## `prefill` 在这里具体是什么意思

在这套导出里，`prefill` 表示：

- 一次输入完整上下文序列
- 当前主链激活张量是 `seq_len=8192`
- attention 计算直接在这整段序列上展开，而不是像 `decode` 一样只处理单 token

因此读这个目录时，最重要的直觉应该是：

- 这是一套“整段上下文一次性灌入”的图
- 不是单 token 递推图

## 主链里哪些图最重要

对大多数读者，优先关注下面四张图就够了：

### `layer_00_linear_attn_block_8k.onnx`

- 第 0 层 merged linear attention block
- 等价于 `input_norm -> DeltaNet(prefill) -> residual add`
- 主干输入输出都是 `seq_len=8192`

### `layer_00_moe_block_8k.onnx`

- 第 0 层 merged MoE block
- 等价于 `post_norm -> moe_ffn -> residual add`

### `layer_03_full_attn_block_8k.onnx`

- 第 3 层 merged full attention block
- 等价于 `input_norm -> rotary -> self_attn(prefill) -> residual add`
- 这是当前目录里最直接表达“整段 8k prefill”语义的主图

### `layer_03_moe_block_8k.onnx`

- 第 3 层 merged MoE block
- 等价于 `post_norm -> moe_ffn -> residual add`

## 这些主图之间怎么理解

这里有两条**并列的代表层样本路径**，不是一条要顺序首尾相连执行的局部网络：

### `layer 0` 样本路径

`embedding_8k -> layer_00_linear_attn_block_8k -> layer_00_moe_block_8k`

### `layer 3` 样本路径

`layer_03_full_attn_block_8k -> layer_03_moe_block_8k`

重点是：

- `layer_00_*` 和 `layer_03_*` 是并列代表样本
- 不是“当前目录里先跑完 `layer_00_*` 再去接 `layer_03_*`”

## 你真正需要关心的 `prefill` 差异

在 `prefill` 目录里，最重要的区别是：

- 主链 `hidden_states` 都是 `seq_len=8192`
- 不再依赖 decode 风格的单步历史 cache 接口
- linear attention 的 recurrence 仍然存在，但发生在整段 chunked prefill 计算内部

其中两类 attention 的表达方式不同：

### full attention

`layer_03_full_attn_block_8k.onnx` 直接在图里表达整段 `8192` token 的 prefill attention 计算。

### linear attention

`layer_00_linear_attn_block_8k.onnx` 顶层图中包含一个 `qwen_onnx::ChunkGatedDeltaRule`，表示 prefill 下的 chunked DeltaNet 递推。

所以 linear attention 的 prefill 差异，主要体现在：

- 顶层 custom op：`ChunkGatedDeltaRule`
- 参考展开图：`layer_00_linear_attn_block_ChunkGatedDeltaRule_chunk64_8k.onnx`
- 更细一步的 `ChunkStep` 参考图：`layer_00_linear_attn_block_ChunkGatedDeltaRule_DeltaNetChunkStep_chunk64.onnx`

## custom op 参考子图是什么

### `layer_00_linear_attn_block_ChunkGatedDeltaRule_chunk64_8k.onnx`

这个文件的意义是：

- 它不是主链上的独立 block
- 它是顶层 `layer_00_linear_attn_block_8k.onnx` 中 `qwen_onnx::ChunkGatedDeltaRule` 的参考展开图
- 主要用于：
  - 结构分析
  - custom op 对照
  - 流程表细看内部实现

### `layer_00_linear_attn_block_ChunkGatedDeltaRule_DeltaNetChunkStep_chunk64.onnx`

这个文件进一步把 `ChunkGatedDeltaRule` 里的单 chunk 递推步骤拆开：

- 对应源码 `torch_chunk_gated_delta_rule()` 的 chunk loop 内部计算
- 内部数值路径是 `FLOAT`
- 主要用于核对：
  - `k_cumdecay_i @ recurrent_state`
  - `attn_inter`
  - `new_recurrent_state`

## 逐文件说明

下面按“主链图优先、辅助图后置”的顺序说明。

### `embedding_8k.onnx`

- 作用：token embedding
- 输出：`hidden_states:[1, 8192, 2048]`

### `layer_00_linear_attn_block_8k.onnx`

- 作用：第 0 层 merged linear attention block
- 输入：
  - `hidden_states:bf16:[1, 8192, 2048]`
  - `conv_state:bf16:[1, 8192, 4]`
  - `recurrent_state:float:[1, 32, 128, 128]`
  - `padding_mask:bool:[1, 8192]`
- 输出：
  - `hidden_states:bf16:[1, 8192, 2048]`
  - `new_conv_state:bf16:[1, 8192, 4]`
  - `new_recurrent_state:float:[1, 32, 128, 128]`
- 说明：
  - 已经融合 `input_norm`
  - 顶层图中包含一个 `qwen_onnx::ChunkGatedDeltaRule`
  - `hidden_states` 主输出回到模型主 dtype
  - `new_recurrent_state` 保持 `FLOAT`，和源码 recurrence 语义一致

### `layer_00_moe_block_8k.onnx`

- 作用：第 0 层 merged MoE block
- 输入：主干输入 `hidden_states`
- 输出：`hidden_states`

### `layer_03_full_attn_block_8k.onnx`

- 作用：第 3 层 merged full attention block
- 输入：整段 `8k` hidden states / position_ids / attention_mask
- 输出：整段 `8k` hidden states
- 说明：
  - 这是 prefill full attention 的代表图
  - 不走 decode 风格 `past_key/past_value` 单步更新接口

### `layer_03_moe_block_8k.onnx`

- 作用：第 3 层 merged MoE block
- 输入：主干输入 `hidden_states`
- 输出：`hidden_states`

### `norm_8k.onnx`

- 作用：final norm
- 输入：`hidden_states:[1, 8192, 2048]`
- 输出：`output:[1, 8192, 2048]`

### `lm_head_8k.onnx`

- 作用：lm head
- 输入：`hidden_states:[1, 8192, 2048]`
- 输出：`logits:[1, 8192, vocab_size]`

### `layer_00_linear_attn_block_ChunkGatedDeltaRule_chunk64_8k.onnx`

- 作用：`ChunkGatedDeltaRule` custom op 参考展开图
- 输入：
  - `query/key/value/g/beta:bf16`
  - `recurrent_state:float`
- 输出：
  - `core_out:bf16`
  - `new_recurrent_state:float`
- 说明：
  - 对外接口遵循“主输出回主 dtype、state 保持 float”的源码语义
  - 图内包含 `128` 个 `qwen_onnx::DeltaNetChunkStep`

### `layer_00_linear_attn_block_ChunkGatedDeltaRule_DeltaNetChunkStep_chunk64.onnx`

- 作用：`DeltaNetChunkStep` 参考子图
- 输入：
  - `q_i/k_i/v_i/decay_i/g_i/g_last/k_cumdecay_i/recurrent_state:float`
  - `upper_mask:bool`
- 输出：
  - `core_out:float`
  - `new_recurrent_state:float`
- 说明：
  - 这一层对应源码 chunk loop 体内部计算
  - 内部和输出都保持 `FLOAT`
  - 主要用于 recurrence 内部计算流对照

## 目录里的辅助文件

- `layer_00_linear_attn_block_8k.json`
  - structured manifest
  - 用来描述顶层主图和 `ChunkGatedDeltaRule` / `DeltaNetChunkStep` 参考图之间的关系
- `onnx_stats.json`
  - 每个导出文件的基础统计
- `onnx_flow_stats_multi.xlsx`
  - 批量流程/算子统计报表
- `Qwen3_5_MoE.jpg`
  - 结构示意图
