# Hugging Face DLSA

![Version: 0.1.0](https://img.shields.io/badge/Version-0.1.0-informational?style=flat-square) ![Type: application](https://img.shields.io/badge/Type-application-informational?style=flat-square) ![AppVersion: 1.16.0](https://img.shields.io/badge/AppVersion-1.16.0-informational?style=flat-square)

A Helm chart for Kubernetes

## Requirements

| Repository | Name | Version |
|------------|------|---------|
| https://argoproj.github.io/argo-helm | argo-workflows | 0.20.6 |

## Values

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| dataset.key | string | `"sst"` |  |
| kubernetesClusterDomain | string | `"cluster.local"` |  |
| metadata.name | string | `"hugging-face-dlsa"` |  |
| proxy | string | `"nil"` |  |
| volumeClaimTemplates.output_dir.resources.requests.storage | string | `"1Gi"` |  |
| volumeClaimTemplates.workspace.resources.requests.storage | string | `"2Gi"` |  |
| workflow.dataset | string | `"sst2"` |  |
| workflow.model | string | `"bert-large-uncased"` |  |
| workflow.num_nodes | int | `2` | \# Nodes |
| workflow.process_per_node | int | `2` | \# Instances |
| workflow.ref | string | `"v1.0.0"` |  |
| workflow.repo | string | `"https://github.com/intel/document-level-sentiment-analysis"` |  |

