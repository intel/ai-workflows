# Document Level Sentiment Analysis

![Version: 0.1.0](https://img.shields.io/badge/Version-0.1.0-informational?style=flat-square) ![Type: application](https://img.shields.io/badge/Type-application-informational?style=flat-square) ![AppVersion: 1.16.0](https://img.shields.io/badge/AppVersion-1.16.0-informational?style=flat-square)

A Helm chart for Kubernetes

## Values

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| dataset.datasetKey | string | `"sst"` | path to dataset; either `sst` or `aclImdb` |
| dataset.logsKey | string | `"sst"` | path to save output logs |
| dataset.nfs.path | string | `"nil"` | path to nfs share |
| dataset.nfs.readOnly | bool | `true` |  |
| dataset.nfs.server | string | `"nil"` |  |
| dataset.nfs.subPath | string | `"nil"` | subpath to dataset directory in nfs share |
| dataset.type | string | `"nfs"` | either `nfs` or `s3` |
| metadata.name | string | `"document-level-sentiment-analysis"` |  |
| proxy | string | `"nil"` |  |
| volumeClaimTemplates.output_dir.resources.requests.storage | string | `"1Gi"` |  |
| volumeClaimTemplates.workspace.resources.requests.storage | string | `"2Gi"` |  |
| workflow.dataset | string | `"sst2"` | `sst2` for the `sst` dataset or `imdb` for the `aclImdb` dataset |
| workflow.model | string | `"bert-large-uncased"` | Model name on Hugging Face |
| workflow.num_nodes | int | `2` | \# of Nodes |
| workflow.process_per_node | int | `2` | \# of Instances Per Node |
| workflow.ref | string | `"v1.0.0"` |  |
| workflow.repo | string | `"https://github.com/intel/document-level-sentiment-analysis"` |  |

----------------------------------------------
Autogenerated from chart metadata using [helm-docs v1.11.0](https://github.com/norwoodj/helm-docs/releases/v1.11.0)
