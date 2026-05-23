# TelaRec

> A Thermoelastic Wave-based Sequential Recommendation model built on RecBole.

---

## 📊 Benchmark Datasets

This project evaluated the **TelaRec** model on the following benchmark datasets:
* **Amazon_ratings** (Rating-only categories):
  * **Beauty**
  * **Sports & Outdoors** 
  * **Video Games**
  * **Electronics** 
* **Gowalla** (Location-based social network dataset):
  * **Gowalla-Merged** 
* **MovieLens** (Popular movie recommendation benchmark dataset):
  * **MovieLens-1M**

### Dataset Download & Placement

You can obtain the preprocessed atomic files (`.inter`) for these datasets using the following methods:

#### 1. Automatic Download
RecBole supports automatic downloading for several standard benchmark datasets. When you run a command using these datasets for the first time, they will be automatically downloaded and extracted into the `dataset/` directory.

#### 2. Manual Download from Official Channels
You can manually download the preprocessed atomic files from the following official channels:
* **GitHub Repository**: [RUCAIBox/RecDatasets](https://github.com/RUCAIBox/RecDatasets)
* **Google Drive**: [Processed Datasets in Google Drive](https://drive.google.com/drive/folders/1so0lckI6N6_niVEYaBu-LIcpOdZf99kj?usp=sharing)
* **Baidu Wangpan**: [Baidu Wangpan Link](https://pan.baidu.com/s/1p51sWMgVFbAaHQmL4aD_-g) (Extraction Code / Password: `e272`)

> [!TIP]
> If you download the datasets manually, please ensure they are extracted and placed under the `dataset/` folder in the project root (e.g., `dataset/beauty/`, `dataset/gowalla-m/`) so the program can locate them correctly.

---

## 🚀 How to Run TelaRec

To train and evaluate the **TelaRec** model on a specific dataset, execute the `run_recbole.py` script from the project root directory:

```bash
python run_recbole.py --model=TelaRec --dataset=<dataset_name>
```

### Examples

* **Run on Amazon Beauty dataset**:
  ```bash
  python run_recbole.py --model=TelaRec --dataset=beauty
  ```

* **Run with custom hyperparameters** (e.g., modifying learning rate and embedding size):
  ```bash
  python run_recbole.py --model=TelaRec --dataset=beauty --learning_rate=0.001 --embedding_size=64
  ```

### ⚙️ Configuration & Optimal Hyperparameters

The default hyperparameters for TelaRec are stored in the configuration file:
* Configuration File Path: [TelaRec.yaml](recbole/properties/model/TelaRec.yaml)

#### Optimal Hyperparameters for Benchmark Datasets

Here are the optimal settings for `c_init`, `alpha_init`, and `kappa_init` extracted from [TelaRec.yaml](recbole/properties/model/TelaRec.yaml). You can run them by passing the hyperparameters directly in the command:

| Dataset | `c_init` | `alpha_init` | `kappa_init` | Example Run Command |
| :--- | :---: | :---: | :---: | :--- |
| **Beauty** | `1.0` | `0.001` | `0.1` | `python run_recbole.py --model=TelaRec --dataset=beauty --c_init=1.0 --alpha_init=0.001 --kappa_init=0.1` |
| **Sports** | `0.5` | `0.0001` | `0.5` | `python run_recbole.py --model=TelaRec --dataset=sports --c_init=0.5 --alpha_init=0.0001 --kappa_init=0.5` |
| **Video** | `0.1` | `0.001` | `0.5` | `python run_recbole.py --model=TelaRec --dataset=video --c_init=0.1 --alpha_init=0.001 --kappa_init=0.5` |
| **Elec** | `0.5` | `0.01` | `0.01` | `python run_recbole.py --model=TelaRec --dataset=elec --c_init=0.5 --alpha_init=0.01 --kappa_init=0.01` |
| **Gowalla-M** | `0.1` | `0.01` | `0.001` | `python run_recbole.py --model=TelaRec --dataset=gowalla-m --c_init=0.1 --alpha_init=0.01 --kappa_init=0.001` |
| **ML-1M** | `1.0` | `0.001` | `0.001` | `python run_recbole.py --model=TelaRec --dataset=ml-1m --c_init=1.0 --alpha_init=0.001 --kappa_init=0.001` |