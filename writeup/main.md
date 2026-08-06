# Agent-Perf: LLM Inference Serving Under Agentic Workloads

> Study overview. Chapter reports in ch1.md through ch4.md.

## Thesis
<!-- OWNER WRITES: one-paragraph summary of key finding after all chapters complete -->

## Methodology
See [METHODOLOGY.md](../METHODOLOGY.md). All 8 rules followed for every reported result.

## Chapter 1: Frameworks Under Agent Traffic
<!-- OWNER WRITES: one-sentence headline finding -->
[Full chapter](ch1.md) | [Run script](../chapters/ch1_frameworks/run.sh)

## Chapter 2: The Precision Ladder
<!-- OWNER WRITES: headline finding for the sm_120 FP4 data -->
[Full chapter](ch2.md) | [Run script](../chapters/ch2_quantization/run.sh)

## Chapter 3: Scheduler & Memory Knobs
<!-- OWNER WRITES: headline finding or surprising interaction -->
[Full chapter](ch3.md) | [Run script](../chapters/ch3_scheduler_knobs/run.sh)

## Chapter 4: Tensor Parallelism over PCIe Gen5
<!-- OWNER WRITES: FP8-single vs TP2-BF16 verdict -->
[Full chapter](ch4.md) | [Run script](../chapters/ch4_tensor_parallel/run.sh)

## The Non-Obvious Finding
<!-- OWNER WRITES: the most surprising result -->

## Limitations
- Single-machine, workstation-class GPU (PCIe Gen5, no NVLink)
- Version snapshot; see ENVIRONMENTS.md for exact versions
- Quality evaluation limited to 50-item programmatic task pack

## Upstream Contributions
See [CONTRIBUTIONS.md](../CONTRIBUTIONS.md).
