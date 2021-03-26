import numpy as np

import os
import sys
import re
from glob import glob
import time
import argparse

from machines import VirtualMachine, PhysicalMachine


def debug(info: object = '', linesep: str = os.linesep) -> None:
    """
    调试信息
    ======

    :param info: 信息内容
    :param linesep: 断行符号
    :return: 无
    """

    # 非debug模式下不输出
    if not args.verbose: return

    if isinstance(info, str):
        # 对非空输出行加前缀
        line = ''.join(['[Debug] ' if info != '' else '', info, linesep])

        sys.stderr.write(line)
        sys.stderr.flush()

    elif isinstance(info, dict):
        for k, v in info.items():
            line = ''.join(['[Debug] ', f'{k}: {v}', linesep])
            sys.stderr.write(line)
            sys.stderr.flush()


def react(info: str, linesep=os.linesep):
    """
    输出信息
    ======

    和裁判系统的操作信息

    :param info: 信息内容
    :param linesep: 断行符号
    :return: 无
    """

    info += linesep
    sys.stdout.write(info)
    sys.stdout.flush()


class Monitor:
    def __init__(self, T):
        # 记录所有正在运行的物理机
        self.running_physical_machines = []

        # 记录虚拟机节点所在的物理机的id
        self.virtual_physical_mapping = {}

        # 总成本
        self.cost = 0

        # 开始时间
        self.start_time = time.time()

        # 扩容需求
        self.demands = {}

        # 部署任务
        self.commands = []

        # 每天的分配候选机器
        self.candidates = []

        # 总天数
        self.T = T

    @staticmethod
    def consumption(pm: PhysicalMachine):
        free_resources = np.array(pm.get_free_resources())
        p = 1 / np.array(pm.A.volume())
        used = 1 - (free_resources * p)
        return used

    def best_physical_machine(self):
        return list(possible_physical_machines.keys())[0]

    def try_add_virtual_machine(self, model: str, idx):
        """
        来自用户的请求，新增虚拟机
        ======================

        :param model: 虚拟机型号
        :param idx: 虚拟机节点的id
        :return:
        """

        done = False

        """
        可以优化的点
        ==========

        目前是遍历一遍所有的机器看哪个机器足够创建服务器，
        一旦物理机太多就会变得很慢，
        最好能以天为单位求最优解
        """
        vm = VirtualMachine(model,
                            possible_virtual_machines[model]['cpu'],
                            possible_virtual_machines[model]['ram'],
                            possible_virtual_machines[model]['double_type'])

        for i, pm in enumerate(self.running_physical_machines):
            result = pm.try_add_virtual_machines(vm, idx)
            done = (result != '')
            if done:
                # 记录虚拟机节点所在的物理机的id
                self.virtual_physical_mapping[idx] = i
                # 记录部署结果
                self.commands.append("".join([f"({i}", f", {result})" if result != "AB" else ")"]))
                break

        """
        可以优化的点
        ==========
        
        如果还不能创建那就再买
        """
        if not done:
            # 确定有物理机可以买，且已买的物理机数量不超过10^5
            if not len(possible_physical_machines) > 0 or not (0 <= len(self.running_physical_machines) < 1e5):
                return done

            """
            可以优化的点
            ==========

            目前是直接买容量最大的服务器（实际上可能有cpu大但ram小、cpu小但ram大的例子，买不到最优解）
            """
            self.record_daily_demands(self.best_physical_machine(), Q=1)

            result = self.running_physical_machines[-1].try_add_virtual_machines(vm, idx)
            done = (result != '')
            if done:
                # 物理机id
                i = len(self.running_physical_machines) - 1
                # 记录虚拟机节点所在的物理机的id
                self.virtual_physical_mapping[idx] = i
                # 记录部署结果
                self.commands.append("".join([f"({i}", f", {result})" if result != "AB" else ")"]))

        return done

    def record_daily_demands(self, model: str, Q: int = 1):
        """
        购买 Q 台指定型号的物理机
        =====================

        :param model:
        :param Q: 购买数量
        :return:
        """
        assert model in possible_physical_machines.keys()

        # 提出需要 Q 台 model 型号的物理机
        if model in self.demands.keys():
            self.demands[model] += Q
        else:
            self.demands[model] = Q

        # 增加 Q 台物理机
        for q in range(Q):
            pm = PhysicalMachine(model,
                                 possible_physical_machines[model]['cpu'],
                                 possible_physical_machines[model]['ram'],
                                 possible_physical_machines[model]['fixed_cost'],
                                 possible_physical_machines[model]['daily_cost'])
            # 记录已有的物理机
            self.running_physical_machines.append(pm)
            # 计算固定成本
            self.cost += pm.fixed_cost

    def del_virtual_machine(self, idx):
        """
        来自用户的请求，删除虚拟机
        ======================

        :param idx: 虚拟机节点的id
        :return:
        """

        # 找到虚拟机节点所在的物理机
        i = self.virtual_physical_mapping[idx]
        pm = self.running_physical_machines[i]
        # 删除虚拟机节点
        pm.del_virtual_machines(idx)
        # 删除记录
        self.virtual_physical_mapping.pop(idx)

        return True

    def migrate(self, t):
        # limit = len(self.virtual_physical_mapping) * 5 // 1000
        # pms = list(sorted(self.running_physical_machines,
        #                   key=lambda x: np.array(x.get_free_resources()).sum(),
        #                   reverse=False))
        #
        # debug(f"{t:>3}-th day's storage information")
        # [info(pm) for i, pm in enumerate(pms) if i >= 0]
        # if limit < 1 or len(pms) < 1:
        #     debug("No vm can be migrated today")
        # debug()
        #
        # if t > 10: exit(0)
        pass

    def purchase_physical_machines_daily(self):
        """
        每日扩容
        ======

        :return:
        """

        react(f'(purchase, {len(self.demands)})')
        for model, Q in self.demands.items():
            react(f'({model}, {Q})')
        self.demands.clear()

    def migrate_virtual_machines_daily(self):
        """
        虚拟机迁移，核心算法
        =================

        在处理完每一天的所有操作后(包括迁移，创建和删除)，
        裁判程序会将当前有负载(至少部署了一台虚拟机)的服务器视为开机状态，
        没有任何负载的服务器视为关机状态

        :return:
        """

        self.cost += 0

        # 不迁移机器（迁移 0 台机器）
        react(f'(migration, 0)')

    def deploy_virtual_machines_daily(self):
        """
        每日节点部署
        ==========

        :return:
        """

        for cmd in self.commands:
            react(cmd)
        self.commands.clear()

    def power_off_physical_machines_daily(self, t=0):
        """
        对没有负载的服务器关机
        ==================

        在处理完每一天的所有操作后(包括迁移，创建和删除)，
        裁判程序会将当前有负载(至少部署了一台虚拟机)的服务器视为开机状态，
        没有任何负载的服务器视为关机状态
        """

        for day in range(t):
            mask = [False] * len(self.running_physical_machines)

            for i, pm in enumerate(self.running_physical_machines):
                mask[i] = (len(pm.running_virtual_machiness) > 0)

            self.running_physical_machines = np.array(self.running_physical_machines)[mask].tolist()

    def calculate_cost_daily(self, t):
        """
        计算总能耗
        ========

        :param t: index of day
        :return:
        """

        num_running_physical_machines = 0
        consumptions = []

        last_day = (t == self.T - 1)

        num_running_physical_machines += len(self.running_physical_machines)
        for pm in self.running_physical_machines:
            if len(pm.get_virtual_machines()) > 0:
                self.cost += pm.daily_cost
                if last_day:
                    consumptions.append(self.consumption(pm))

        debug()
        debug(f'{t:>4d}-th day\'s cost = {self.cost:,}')
        debug(f'         Time cost = {time.time() - self.start_time:.3f}s')
        debug(f'  Running machines : pm = {num_running_physical_machines} \t vm = {len(self.virtual_physical_mapping)}')
        if last_day:
            consumptions = np.array(consumptions)
            consumptions = consumptions.mean(axis=0)
            debug(f"       Consumption : CPU-A {consumptions[0][0]:<5.1%} | RAM-A {consumptions[0][1]:<5.1%}")
            debug(f"                     CPU-B {consumptions[1][0]:<5.1%} | RAM-B {consumptions[1][1]:<5.1%}")
        debug()


class Dataset:
    def __init__(self, filename=None):
        self.data = None
        if filename is not None and os.path.exists(filename):
            self.data = open(filename, 'r', encoding='utf-8').readlines()
        self.cursor = 0

    def __getitem__(self, item):
        if self.data is None:
            return input()
        return self.data[item]

    def pop(self):
        if self.data is None:
            return input()
        line = self.data[self.cursor]
        self.cursor += 1
        return line


def process(dataset: Dataset):
    global possible_physical_machines, possible_virtual_machines

    # 可以采购的服务器类型和数量
    # ======================

    # 第一行包含一个整数 N(1≤N≤100)，表示可以采购的服务器类型数量。
    N = int(dataset.pop())
    # 接下来 N 行，每行描述一种类型的服务器，数据格式为：(型号, CPU 核数, 内存 大小, 硬件成本, 每日能耗成本)。
    # 例如(NV603, 92, 324, 53800, 500)表示一种服务器类型，其型号为 NV603，包含 92 个 CPU 核心，324G 内存，
    # 硬件成本为 53800，每日能耗成本为 500。CPU 核数，内存大小，硬件成本，每日能耗成本均为正整数。
    # 每台服务器的 CPU 核数以及内存大小不超过 1024，硬件成本不超过 5 × 105 ，每日能耗成本不超过 5000。
    # 服务器型号长度不超过 20，仅由数字和大小写英文字符构成。
    for n in range(N):
        line = re.sub('[ ()]', '', dataset.pop().strip())

        model, cpu, ram, fixed_cost, daily_cost = line.split(',')
        possible_physical_machines[model] = {'cpu': int(cpu),
                                             'ram': int(ram),
                                             'fixed_cost': int(fixed_cost),
                                             'daily_cost': int(daily_cost)}
        # PhysicalMachine(model, int(cpu), int(ram), int(fixed_cost), int(daily_cost))
        # ==> end of N

    # # 以 cpu - ram 作为关键特征1，以 cpu+ram 的单价为关键特征2，升序
    # # 得出性能最均衡的、性价比最高的服务器
    # ===================================================
    # [Debug] Numa A:  138472 C  |  43340 G
    # [Debug] Numa B:  200882 C  |  109420 G
    #
    # [Debug]  799-th day's cost = 863342535
    # [Debug]          Time cost = 27.924s
    # [Debug]   Running machines : pm = 5511   vm = 24427
    #
    # [Debug] cost time: 27.923732042312622
    # ===================================================
    # possible_physical_machines = dict(
    #     sorted(possible_physical_machines.items(),
    #            key=lambda x: (abs(x[1]['cpu'] - x[1]['ram']),
    #                           (x[1]['fixed_cost'] + (10 * x[1]['daily_cost'])) / (x[1]['cpu'] + x[1]['ram'])),
    #            reverse=False)
    # )

    # 以 cpu - ram 作为关键特征1，以 cpu+ram 的单价为关键特征2，升序
    # 得出性能最均衡的、配置最高的服务器
    # ===================================================
    # 优先放入A（不管剩余空间大小）
    #
    # [Debug] Numa A:  123684 C  |  27778 G
    # [Debug] Numa B:  160769 C  |  78543 G
    #
    # [Debug]  799-th day's cost = 887547748
    # [Debug]          Time cost = 14.496s
    # [Debug]   Running machines : pm = 2281   vm = 24427
    #
    # [Debug] cost time: 14.49584412574768
    # ===================================================
    # 加入 Numa A 和 Numa B 切换的机制 (优先放入大的）
    #
    # [Debug] Numa A:  52383 C  |  27021 G
    # [Debug] Numa B:  20987 C  |  37892 G
    #
    # [Debug]  799-th day's cost = 1265328689
    # [Debug]          Time cost = 29.803s
    # [Debug]   Running machines : pm = 3372   vm = 24427
    #
    # [Debug] cost time: 29.803368091583252
    # ===================================================
    # 加入 Numa A 和 Numa B 切换的机制 (优先放入小的）
    #
    # [Debug] Numa A:  114123 C  |  34340 G
    # [Debug] Numa B:  297294 C  |  55824 G
    #
    # [Debug]  799-th day's cost = 943866101
    # [Debug]          Time cost = 19.243s
    # [Debug]   Running machines : pm = 2450   vm = 24427
    #
    # [Debug] cost time: 19.242738962173462
    # ===================================================
    possible_physical_machines = dict(
        sorted(possible_physical_machines.items(),
               key=lambda x: (abs(x[1]['cpu'] - x[1]['ram']),
                              -1 * (x[1]['cpu'] + x[1]['ram'])),
               reverse=False)
    )

    # 配置最高 （CPU + RAM） 最大的服务器
    # ===================================================
    # [Debug] Numa A:  342055 C  |  27855 G
    # [Debug] Numa B:  349660 C  |  72062 G
    #
    # [Debug]  799-th day's cost = 1026732026
    # [Debug]          Time cost = 14.845s
    # [Debug]   Running machines : pm = 2256   vm = 24427
    #
    # [Debug] cost time: 14.844807147979736
    # ===================================================
    # possible_physical_machines = dict(sorted(possible_physical_machines.items(), key=lambda x: x[1]['cpu'] + x[1]['ram'], reverse=True))

    debug()
    debug(f'Possible Physical Machines')
    debug(f'==========================\n')
    debug(possible_physical_machines)

    # 可供售卖的虚拟机类型和数量
    # ======================

    # 接下来一行包含一个整数 M(1≤M≤1000)，表示售卖的虚拟机类型数量。
    M = int(dataset.pop())
    # 接下来 M 行，每行描述一种类型的虚拟机，数据格式为：(型号, CPU 核数, 内存大小, 是否双节点部署)。
    # 是否双节点部署用 0 和 1 表示，0 表示单节点部署，1 表示双节点部署。
    #   例如：
    #     (s3.small.1, 1, 1, 0) 表示一种虚拟机类型，其型号为 s3.small.1，所需 CPU 核数为 1，所需内存为 1G，单节点部署；
    #     (c3.8xlarge.2, 32, 64, 1) 表示一种虚拟机类型，其型号为 c3.8xlarge.2，所需 CPU 核数为 32，所需内存为 64G，双节点部署。
    # CPU 核数，内存大小均为正整数。
    # 对于每种类型的虚拟机，数据集保证至少存在一种服务器可以容纳。虚拟机型号长度不超过 20，仅由数字，大小写英文字符和'.'构成。
    for m in range(M):
        line = re.sub('[ ()]', '', dataset.pop().strip())

        model, cpu, ram, double_type = line.split(',')
        possible_virtual_machines[model] = {'cpu': int(cpu),
                                            'ram': int(ram),
                                            'double_type': (double_type == '1')}
        # VirtualMachine(model, int(cpu), int(ram), bool(double_type))
        # ==> end of M

    debug()
    debug(f'Possible Virtual Machines')
    debug(f'=========================\n')
    debug(possible_virtual_machines)

    # 每日操作
    # ======

    # 接下来一行包含一个整数 T(1≤T≤1000)，表示题目共会给出 T 天的用户请求序列数据。
    T = int(dataset.pop())

    monitor = Monitor(T)

    debug()
    debug(f'Daily operations')
    debug(f'================\n')

    # 接下来会按顺序给出 T 天的用户请求序列
    for t in range(T):

        """
        1. 记录需求
        """

        # monitor.migrate(t)

        # 对于每一天的数据，第一行包含一个非负整数 R 表示当天共有 R 条请求。
        R = int(dataset.pop())

        # 接下来 R 行，按顺序给出每一条请求数据。请求数据的格式为：
        #     (add, 虚拟机型号, 虚拟机 ID)    创建一台虚拟机
        #     (del, 虚拟机 ID)              删除一台虚拟机
        #   例如：
        #     (add, c3.large.4, 1)表示创建一台型号为 c3.large.4，ID 为 1 的虚拟机；
        #     (del, 1)表示删除 ID 为 1 的虚拟机。
        # 虚拟机的 ID 均为整数，每个创建请求的虚拟机 ID 唯一，范围不超过带符号 32 位整数表示的范围。
        # 对于删除操作，数据集保证对应 ID 的虚拟机一定存在。
        # 用户创建请求数量总数不超过 10^5 。
        for r in range(R):
            line = re.sub('[ ()]', '', dataset.pop().strip())

            info = line.split(',')

            if len(info) >= 3 and info[0] == 'add':
                model, idx = info[1:]
                monitor.try_add_virtual_machine(model, idx)
            elif len(info) >= 2 and info[0] == 'del':
                idx = info[-1]
                monitor.del_virtual_machine(idx)
            else:
                pass
            # ==> end of the r-th request

        """
        2. 扩容物理机
        """

        monitor.purchase_physical_machines_daily()

        """
        3. 存量虚拟机的迁移
        """

        # 在完成扩容后，在处理每一天的新请求之前，你还可以对当前存量虚拟机进行一次迁移，即把虚拟机从一台服务器迁移至另一台服务器。
        # 对于单节点部署的虚拟机，将其从一台服务器的 A 节点迁移至 B 节点(或反之)也是允许的。
        # 迁移的目的服务器和节点必须有足够的资源容纳所迁移的虚拟机。
        # 迁移的虚拟机总量不超过当前存量虚拟机数量的千分之五。
        # 即假设当前有 n 台存量虚拟机，每天你可以迁移的虚拟机总量不得超过 5n/1000 向下取整。
        monitor.migrate_virtual_machines_daily()

        """
        4. 部署虚拟机节点
        """

        # 执行1阶段中做过以便的部署任务
        monitor.deploy_virtual_machines_daily()

        # 裁判程序会将当前有负载(至少部署了一台虚拟机)的服务器视为开机状态，没有任何负载的服务器视为关机状态
        # monitor.power_off_physical_machines_daily()

        """
        5. 成本核算
        """

        # 计算新的运营费用
        # 总成本包含两部分：购买服务器的整体硬件成本以及服务器消耗的整体能耗成本。
        # 整体硬件成本即将选手输出的方案中所有购买的服务器的硬件成本相加。
        # 整体能耗成本的计算方式为：在处理完每一天的所有操作后(包括迁移，创建和删除)

        # 只有测试算法的时候才运行这一步
        if args.verbose: monitor.calculate_cost_daily(t)

        # ==> end of the t-th day
    debug(f'cost time: {time.time() - monitor.start_time}')


def main():
    # 获取所有线下数据集的文件名
    trainings = sorted(glob(os.path.join(os.path.abspath(os.path.dirname(__file__)), '..', '..',
                                         'training-data',
                                         'training-[0-9].txt')))

    # 初始化数据集
    if args.dataset is not None:
        dataset = Dataset(trainings[int(args.dataset)])
    else:
        dataset = Dataset()

    # 开始干正事
    process(dataset)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-d', '--dataset', default=None,
                        help='Index of Datasets, default: None (means using stdin as input)')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print debug information, default: Not')

    parser.set_defaults(verbose=False)
    args = parser.parse_args()

    try:
        args.dataset = int(args.dataset)
    except (ValueError, TypeError):
        debug(f'Cannot convert the Index of Datasets(=\'{args.dataset}\') from string to int, use None instead.')
        args.dataset = None

    # 可供扩容的物理机
    possible_physical_machines = {}
    # 可供请求的虚拟机
    possible_virtual_machines = {}

    main()
