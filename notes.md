
Fix for tensorboard install:
```
pip uninstall tb-nightly tensorboard
pip install tensorboard
```

Training:
```
python basicsr/train.py -opt options/train/EDSR/train_rs_EDSR_Lx4.yml
```

```
python basicsr/train.py -opt options/train/EDSR/train_rs_EDSR_Lx4.yml; python basicsr/train.py -opt options/train/EDSR/train_rs_EDSR_Lx4_2.yml
```

Testing:
```
python basicsr/test.py -opt options/test/EDSR/test_rs_EDSR_Lx4.yml
```