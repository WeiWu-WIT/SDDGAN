# SDDGAN-tensorflow
Sematic-Supervised Infrared and Visible Image Fusion via Dual-Discriminator Generative Adversarial Network

This work can be applied for infrared and visible image fusion

#Framework

![image](https://user-images.githubusercontent.com/77524447/147047095-df887c96-1bd2-4fad-b3ea-d74fd1e68f24.png)

#Architecture of the generator

![image](https://user-images.githubusercontent.com/77524447/147047201-0f3d9e2c-d0d5-49bc-af3f-e790014ddada.png)

#Architecture of the discriminator

![image](https://user-images.githubusercontent.com/77524447/147047224-4538b668-2eff-4838-a82b-723e7b3b6161.png)

If this work is helpful to you, please cite it as:
```
@article{zhou2021semantic,
  title={Semantic-supervised Infrared and Visible Image Fusion via a Dual-discriminator Generative Adversarial Network},
  author={Zhou, Huabing and Wu, Wei and Zhang, Yanduo and Ma, Jiayi and Ling, Haibin},
  journal={IEEE Transactions on Multimedia},
  year={2021},
  publisher={IEEE}
}
```


#To train:
python3 main.py

#To test:
python3 test_one_image.py

Note:
The weight map generate method and evaluate methods are shown in 'matLab_code'. And these methods are implemented by MatLab.

This code is base on the code of FusionGAN.
