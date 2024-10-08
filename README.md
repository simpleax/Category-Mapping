## Category Mapping of Emergency Supplies Classification Standard Based on BERT-TextCNN


### Data description (training data \ testing data)

1. The experimental data are stored in the `data/` directory;
2. The `data/` directory includes: 739 GB/T38565 category data and 5250 GPC category data; 798 manually annotated category mapping data;
   the required training data (*_train.json)\test data (*_test.json) need to be divided, with the ratio: (first divided into training set and test set according to 9:1);

### Configuration

The configuration file is in the 'model/' file directory. Due to the file size limitation, the 'pytorch_model.bin' used by the model is not uploaded to this repository. Please download in the "https://huggingface.co/nghuyong/ernie-3.0-base-zh/tree/main".

Note: If 'pytorch_model.bin' is not downloaded, the model will not run.

#### If you don't have a GPU device available, you can use the free GPU device provided by [Google Colaboratory](https://colab.research.google.com/notebooks/intro.ipynb)

### Citation and Author
If you use our experimental data or code in your work (in paper review), please cite [Category-Mapping](https://github.com/simpleax/Category-Mapping.git)).

If you need to use our published experimental data and code, please contact us first and get consent before using it!

Email: Dr.Zhang, zhangqiuxia268@163.com

You can email me if you have questions.
