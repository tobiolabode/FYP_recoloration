# SPCOLOR: Semantic-prior Guided Exemplar-based Image Colorization
This is the implementation for  [SPCOLOR: Semantic-prior Guided Exemplar-based Image Colorization](https://arxiv.org/abs/2304.06255). The repo also has the other models including Deep exemplar-based video colorization and Grey2Embed for testing across the model. The goal of this final year project is implment a time adjustment technique for the existing SPCOLOUR project. 

This code is ran via CPU, due to technical reasons. That's an area of further research 


# Shorthand used throughout the code and report
ColorVid: Deep exemplar-based video colorization


**testing**

```python
python /content/FYP_recoloration/train_stego_CPU_ONLY.py --val_only VAL_ONLY --data_root_imagenet /content/FYP_recoloration/dataset/coco-2017 --checkpoint_dir '/content/FYP_recoloration/dataset/checkpoints/spcolor/checkpoints/video_moredata_l1/' --val_output_path 'content/FYP_recoloration/output/'

```

**training**

```python
python /content/FYP_recoloration/train_stego_CPU_ONLY.py --data_root_imagenet /content/FYP_recoloration/dataset/coco-2017 --checkpoint_dir '/content/FYP_recoloration/dataset/checkpoints/spcolor/checkpoints/video_moredata_l1/'
```
