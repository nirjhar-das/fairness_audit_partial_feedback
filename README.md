Official Codebase for the paper *Cost Efficient Fairness Audit Under Partial Feedback* by [Nirjhar Das](https://nirjhar-das.github.io/), [Mohit Sharma](https://mohitsharma29.github.io), [Praharsh Nanavati](https://niftynans.github.io/index.html), [Kirankumar Shiragur](https://sites.google.com/view/kiran-shiragur) and [Amit Deshpande](https://www.microsoft.com/en-us/research/people/amitdesh/).


## Installation
The code base has dependency on basic packages listed in [requirements.txt](./requirements.txt). It can be installed via the following command:
```
$ pip install -r requirements.txt 
```

## Usage
This code base implements `RS-Audit` (Algorithm 1 in the paper), Algorithm 3 in the paper and `Exp-Audit` (Algorithm 2 in the paper). All codes are present under the folder [Experiments](./Experiments).


![all_results](./Experiments/Fairness_Audit_Result.png)

### Reference

If you find this work useful in your research, please consider citing it.

~~~bibtex
@article{sawarni2024optimal,
    title={Cost Efficient Fairness Audit Under Partial Feedback},
    author={Das, Nirjhar and Sharma, Mohit and Nanavati, Praharsh and Shiragur, Kirankumar and Deshpande, Amit},
    year={2025},
    eprint={2404.06831},
    archivePrefix={arXiv},
    primaryClass={cs.LG}
}
~~~
