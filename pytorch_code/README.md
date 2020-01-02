We provide Pytorch implementation of the proposed method here. Note that the model provides basic of functionality of the framework. The general code strucutre and

### Prerequisites and Installation
- Python 2.7
- Pytorch 0.4.1.post2(Or Higher)

#### Getting Started
**Clone this repository:**
```bash
git clone git@github.com:TheSouthFrog/stylealign.git
cd stylealign/pytorch_code
```
**Prepare Dataset**
 - You may the original cropped WFLW dataset here [Google Drive](https://drive.google.com/file/d/1nZmjlwVSJxI8_W27LpjPJa6K_M_gmTXS/view?usp=sharing) or [Baidu Drive](https://pan.baidu.com/s/1fzizdQ3FRBRwfJjGRvw8gA).
 - Unzip downloaded files. Remember to change the parameters of ```root_dir``` and ```image_list``` in ```config/exp_0001.yaml``` to your image directory and annotation file, respectively.

#### Running the code:

```bash
CUDA_VISIBLE_DEVICES=gpu_id python train.py --config ./configs/exp_0001.yaml
```

### Contact
If you have any questions, please feel free to contact the authors.
Shengju Qian sjqian@cse.cuhk.edu.hk

### Citation

If you use our code, please consider citing our paper:

```
@inproceedings{qian2019aggregation,
  title={Aggregation via Separation: Boosting Facial Landmark Detector with Semi-Supervised Style Translation},
  author={Qian, Shengju and Sun, Keqiang and Wu, Wayne and Qian, Chen and Jia, Jiaya},
  journal={ICCV},
  year={2019}
}
```
