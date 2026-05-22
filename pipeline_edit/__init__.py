# pipeline_edit package: splits inference.py implementation by responsibility
#   - utils:      shared utilities (Blender invocation, mask construction, voxelization, model loading, visualization)
#   - preprocess: preprocessing pipeline (multi-view/orthographic rendering, feature extraction, latent caching)
#   - edit:       edit/baseline pipeline (FlowEdit, post-processing rendering, result export)
