# Abstract Reasoning with Distracting Features

To appear in NeurIPS 2019. 

[Abstract Reasoning with Distracting Features](http://arxiv.org/abs/1912.00569)

(Kecheng Zheng, Wei Wei and Zheng-jun Zha, 2019)


<div width="20%", height="20%", align="center">
   <img src="https://github.com/zkcys001/distracting_feature/blob/master/git_images/LEN.png"><br><br>
</div>


# Dataset

To download the dataset, please check [chizhang's project page](http://wellyzhang.github.io/project/raven.html#dataset).

# Performance

For details, please check our [paper](http://arxiv.org/abs/1912.00569).


# Dependencies

**Important**
* Python 2.7
* PyTorch
* CUDA and cuDNN


# Usage

## Benchmarking

```
python main.py --net <model name> --datapath <path to the dataset> --rl False --typeloss False
```

#TODO
1. The code for PGM dataset
2. The long-term features (state) in teacher model 

# Citation

If you find the paper and/or the code helpful, please cite us.

```
@inproceedings{zheng2019abstract,
    title={Abstract Reasoning with Distracting Features},
    author={Kecheng Zheng and Zheng-jun Zha and Wei Wei},
    booktitle={Advances in Neural Information Processing Systems},
    year={2019}}
}
```

# Acknowledgement


* [Wild Relational Network](https://github.com/Fen9/WReN)


## Contact 

Please feel free to discuss paper/code through issues or emails.


### License 
[Apache License 2.0](./LICENSE)
