# Huawei CodeCraft 2021

## 基本说明

用git tags进行区分版本

| Tags | 时间点 | 完成情况 |
| :---: | :---: | :---: |
| final | 03-27 19:55 | 最后参考别人的开源，发现最关键的是[第171行](CodeCraft-2021/src/CodeCraft-2021.py#L171)和[第300行](CodeCraft-2021/src/CodeCraft-2021.py#L300)的random，本地案例效果显著提升 |
| onboarded | 03-26 03:30 | [总成本：2103007242，运行时间：26.12，pypy](../../commit/4433923c15380f0ba23c707753490f6fbe38dbe4)|
| base3 | 03-25 17:38 | [避免长列表append，提升效率](../../commit/59b3baebc4c3809a8d2ed237596980894cbdea0b) |
| base2 | 03-24 11:25 | [代码整理，无功能变更](../../commit/ee6810bcd6cbe12905812b5bcfdbaf9df475cba4) |
| base | 03-24 10:17 | 基本完成python版本的框架搭建，输入输出格式暂无发现的异常，提交发现超时 |

> pypy: 如果没有运行环境，先 `source ./environment.sh` ，在 Linux 和 MacOS 上自动配置环境。

## 特殊用法

通过修改 `build_and_run.sh` 脚本和加入 `argsparser` 库，实现了本地脚本运行时附加调试参数且不影响正常提交结果的功能。具体如下：

```shell script
 ➜  CodeCraft21 git:(master) ✗ ./build_and_run.sh -h    

usage: CodeCraft-2021.py [-h] [-d DATASET] [-v]

optional arguments:
  -h, --help            show this help message and exit
  -d DATASET, --dataset DATASET
                        Index of datasets, default: None (means using stdin as
                        input)
  -v, --verbose         Print debug information, default: Not
```

进一步说明：

```
-d[0-9] # 采用线下数据集training-[0-9].txt
-v      # 输出debug信息
```

具体示例：

```shell script
 ➜  CodeCraft21 git:(master) ✗ ./build_and_run.sh -d0   
(purchase, 1)
(NV604, 1)
(migration, 0)
(0, A)
(0, A)
(purchase, 0)
(migration, 0)
(0)
(purchase, 0)
(migration, 0)
(0, A)
```

```shell script
➜  CodeCraft21 git:(master) ✗ ./build_and_run.sh -d0 -v

[Debug] Possible Physical Machines
[Debug] ==========================

[Debug] NV604: {'cpu': 128, 'ram': 512, 'fixed_cost': 87800.0, 'daily_cost': 800.0}
[Debug] NV603: {'cpu': 92, 'ram': 324, 'fixed_cost': 53800.0, 'daily_cost': 500.0}

[Debug] Possible Virtual Machines
[Debug] =========================

[Debug] c3.large.4: {'cpu': 2, 'ram': 8, 'double_type': False}
[Debug] c3.8xlarge.2: {'cpu': 32, 'ram': 64, 'double_type': True}

[Debug] Daily operations
[Debug] ================

(purchase, 1)
(NV604, 1)
(migration, 0)
(0, A)
(0, A)

[Debug]    0-th day's cost = 88600.0
[Debug]          Time cost = 0.000s

(purchase, 0)
(migration, 0)
(0)

[Debug]    1-th day's cost = 89400.0
[Debug]          Time cost = 0.000s

(purchase, 0)
(migration, 0)
(0, A)

[Debug]    2-th day's cost = 90200.0
[Debug]          Time cost = 0.000s

```