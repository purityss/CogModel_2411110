# CCN 2023: Keynote + Tutorial
## Cognitive models of Behavior: Classical and Deep Learning Approaches
### Maria Eckstein, Kevin Miller, Kim Stachenfeld, Zeb Kurth-Nelson

Code for CCN 2023 tutorial.

Colab can be found here:
https://colab.research.google.com/drive/1HMsETdAFzJ2yQou2m-qJZzDlpfxgcrO8?usp=sharing

All exercises can in principle be done in the colab linked above. However, in case there are internet issues, we recommend downloading this github directory with the following command before the tutorial.

```
git clone https://github.com/kstach01/CogModelingRNNsTutorial
```

# Getting started
To execute a "cell", you'll press Shift-Enter.

### Hosted Colab
If you're working from a hosted colab (recommended):
1. File > Save a copy in Drive
2. Connect (top right) > Connect to a hosted runtime (GPU)

### Run locally as a Jupyter Notebook

You can also open the notebook in jupyter notebook or a locally hosted colab. To run locally, you can open a terminal window and download the code with:

```
git clone https://github.com/kstach01/CogModelingRNNsTutorial
```

### Jupyter notebook

To run as a jupyter notebook, you can in the terminal window run the following. Note that much of the formatting will be lost in Jupyter Notebook.

```
pip install notebook
git clone https://github.com/kstach01/CogModelingRNNsTutorial
cd CogModelingRNNsTutorial
jupyter notebook CCN2023_CogModelingRNNsTutorial2023.ipynb
```

### Local colab kernel

To run locally from colab, you can follow the steps above to install jupyter notebook, then run the following:

```
  jupyter notebook \
    --NotebookApp.allow_origin='https://colab.research.google.com' \
    --port=8888 \
    --NotebookApp.port_retries=0
```
This will generate text including a link which will be what you connect the colab to.

You can then go to [https://research.google.com/colaboratory/](https://research.google.com/colaboratory/) and open CCN2023_CogModelingRNNsTutorial2023.ipynb. 

To connect to the kernel, go to the down arrow in the top right corner > "Connect to local runtime" and then copy in the link generated by the command.