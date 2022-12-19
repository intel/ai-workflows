# vision-transfer-learning

![Version: 0.1.0](https://img.shields.io/badge/Version-0.1.0-informational?style=flat-square) ![Type: application](https://img.shields.io/badge/Type-application-informational?style=flat-square) ![AppVersion: 0.1.0](https://img.shields.io/badge/AppVersion-0.1.0-informational?style=flat-square)

A Helm chart for Kubernetes

## Values

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| dataset.key | string | `false` | set to bucket path if using resisc45 dataset |
| kubernetesClusterDomain | string | `"cluster.local"` |  |
| metadata.name | string | `"vision-transfer-learning"` |  |
| proxy | string | `"nil"` |  |
| sidecars.image | string | `"ubuntu:20.04"` |  |
| volumeClaimTemplates.output_dir.resources.requests.storage | string | `"4Gi"` |  |
| volumeClaimTemplates.workspace.resources.requests.storage | string | `"2Gi"` |  |
| workflow.batch_size | int | `32` |  |
| workflow.num_epochs | int | `100` |  |
| workflow.platform | string | `"None"` |  |
| workflow.precision | string | `"FP32"` |  |
| workflow.ref | string | `"v1.0.1"` |  |
| workflow.repo | string | `"https://github.com/intel/vision-based-transfer-learning-and-inference"` |  |
| workflow.script | string | `"colorectal"` |  |

