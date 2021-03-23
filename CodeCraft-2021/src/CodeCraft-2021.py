import numpy as np

import os
import sys
import re
from collections import OrderedDict

DEBUG = False


def debug(info: str = '', linesep=os.linesep):
    info += linesep
    sys.stderr.write(info)
    sys.stderr.flush()


def react(info: str, linesep=os.linesep):
    info += linesep
    sys.stdout.write(info)
    sys.stdout.flush()


class VirtualMachine:
    def __init__(self, model, cpu, ram, double_type):
        """
        :param model: 型号
        :param cpu: CPU 核数
        :param ram: 内存 大小
        :param double_type: 是否双节点部署
        """

        self.model = model
        self.cpu = cpu
        self.ram = ram
        # 直接表明部署节点数
        self.double_type = double_type


class Device:
    def __init__(self, volume):
        """
        CPU 和 RAM 的父类

        :param volume: 总容量
        """
        assert volume % 2 == 0
        self.volume = volume
        self.free = volume


class Numa:
    def __init__(self, cpu, ram):
        """
        Numa 节点

        :param cpu:
        :param ram:
        """
        self.cpu = Device(cpu)
        self.ram = Device(ram)

    def free(self):
        return self.cpu.free, self.ram.free

    def try_allocate(self, cpu, ram):
        if self.cpu.free >= cpu and self.ram.free >= ram:
            return True
        return False


class PhysicalMachine:
    def __init__(self, model, cpu, ram, fixed_cost, daily_cost):
        """
        单台物理机

        :param model: 型号
        :param cpu: CPU 核数
        :param ram: 内存 大小
        :param fixed_cost: 硬件成本
        :param daily_cost: 每日能耗成本
        """

        # 物理机的类型
        self.model = model

        # Numa 架构：目前主流的服务器都采用了非统一内存访问（Numa）架构，
        # 你可以理解为每台服务器内部都存在两个 Numa 节点：A 和 B（下文中提到的节点均指 Numa 节点）。
        # 服务器拥有的资源（CPU 和内存）均匀分布在这两个节点上。
        # 以 NV603 为例，其 A、B 两个节点分别包含 46C 和 162G 的资源。保证服务器的 CPU 核数和内存大小均为偶数。
        self.A = Numa(cpu / 2, ram / 2)
        self.B = Numa(cpu / 2, ram / 2)

        # 服务器成本：数据中心使用每台服务器的成本由两部分构成：硬件成本和能耗成本。
        # 硬件成本是在采购服务器时的一次性支出，能耗成本是后续服务器使用过程中的持续支出。
        # 为了便于计算，我们以天为单位计算每台服务器的能耗成本。
        # 若一台服务器在一天内处于关机状态，则其不需要消耗任何能耗成本，否则我们需要支出其对应的能耗成本。
        self.fixed_cost = fixed_cost
        self.daily_cost = daily_cost

        # 目前正在此物理机上运行的虚拟机节点
        self.running_virtual_nodes = {}

    def try_add_virtual_node(self, model: str, idx: int):
        """
        尝试指派一个虚拟机节点

        :param model:
        :param idx:
        :return:
        """

        vm = VirtualMachine(model,
                            possible_virtual_machines[model]['cpu'],
                            possible_virtual_machines[model]['ram'],
                            possible_virtual_machines[model]['double_type'])

        # 双节点部署指的是一台虚拟机所需的资源（CPU 和内存）必须由一台服务器的两个节点同时提供，
        # 并且每个节点提供总需求资源的一半。
        # 双节点部署的虚拟机保证其 CPU 和内存需求量都是偶数。
        if vm.double_type:
            assert vm.cpu % 2 == 0 and vm.ram % 2 == 0
            cpu, ram = vm.cpu / 2, vm.ram / 2

            if self.A.try_allocate(cpu, ram) and self.B.try_allocate(cpu, ram):
                self.running_virtual_nodes[idx] = vm
                return True

        # 单节点部署指的是一台虚拟机所需的资源（CPU 和内存）完全由主机上的一个节点提供
        else:
            cpu, ram = vm.cpu, vm.ram

            """
            可以优化的点
            ==========
            
            目前是优先分配到 Numa A ，
            可能会导致长期占用 Numa A 但 Numa B 空闲。
            
            可能解决方案
            ==========
            
            1. 全局分配 Numa A 和 Numa B 。
            2. 单物理机内根据 Numa A 和 Numa B 的占用情况来分配。
            """
            if self.A.try_allocate(cpu, ram):
                self.running_virtual_nodes[idx] = vm
                return True
            elif self.B.try_allocate(cpu, ram):
                self.running_virtual_nodes[idx] = vm
                return True

        return False

    def del_virtual_node(self, idx):
        """
        删除给定下标的虚拟机节点

        :param idx:
        :return:
        """

        if idx not in self.running_virtual_nodes.keys():
            return False

        self.running_virtual_nodes.pop(idx)
        return True

    def get_free_resources(self):
        """
        获取物理机的占用情况

        :return: Numa A 的占用情况，Numa B 的占用情况
        """

        return self.A.free(), self.B.free()

    def get_virtual_nodes(self):
        """
        获取所有正在运行的虚拟机节点

        :return: 所有正在运行的虚拟机节点
        """

        return self.running_virtual_nodes


class Monitor:
    def __init__(self):
        # 记录所有正在运行的物理机
        self.running_physical_machines = []

        # 记录虚拟机节点所在的物理机的id
        self.virtual_physical_mapping = {}

        # 总成本
        self.cost = 0

    def buy_physical_machines(self, model: str, Q: int):
        """
        购买 Q 台指定型号的物理机
        :param model:
        :param Q: 购买数量
        :return:
        """
        assert model in possible_physical_machines.keys()

        # 向裁判系统购买 Q 台物理机
        react(f'({model}, {Q})')

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

    def add_virtual_machine(self, model: str, idx):
        """
        来自用户的请求，新增虚拟机

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
        for i, pm in enumerate(self.running_physical_machines):
            if pm.try_add_virtual_node(model, idx):
                done = True
                # 记录虚拟机节点所在的物理机的id
                self.virtual_physical_mapping[idx] = i
                break

        """
        可以优化的点
        ==========
        
        如果还不能创建那就再买（臭方法）
        """
        while not done:
            # 确定有物理机可以买，且已买的物理机数量不超过10^5
            if not len(possible_physical_machines) > 0 or not (0 <= len(self.running_physical_machines) < 1e5):
                break

            """
            可以优化的点
            ==========

            目前是直接买容量最大的服务器（实际上可能有cpu大但ram小、cpu小但ram大的例子，买不到最优解）
            """
            self.buy_physical_machines(list(possible_physical_machines.keys())[0], Q=1)

            if self.running_physical_machines[-1].try_add_virtual_node(model, idx):
                done = True
                # 记录虚拟机节点所在的物理机的id
                self.virtual_physical_mapping[idx] = len(self.running_physical_machines) - 1
                break

        return done

    def del_virtual_machine(self, idx):
        """
        来自用户的请求，删除虚拟机

        :param idx: 虚拟机节点的id
        :return:
        """

        # 找到虚拟机节点所在的物理机
        print(self.virtual_physical_mapping)
        j = self.virtual_physical_mapping[idx]
        pm = self.running_physical_machines[j]
        # 删除虚拟机节点
        pm.del_virtual_node(idx)
        # 删除记录
        self.virtual_physical_mapping.pop(idx)

        return True

    def migration_physical_machine(self):
        """
        虚拟机迁移，核心算法
        =================

        在处理完每一天的所有操作后(包括迁移，创建和删除)，
        裁判程序会将当前有负载(至少部署了一台虚拟机)的服务器视为开机状态，
        没有任何负载的服务器视为关机状态

        :return:
        """

        pass

    def power_off_physical_machine(self):
        """
        对没有负载的服务器关机
        ==================

        在处理完每一天的所有操作后(包括迁移，创建和删除)，
        裁判程序会将当前有负载(至少部署了一台虚拟机)的服务器视为开机状态，
        没有任何负载的服务器视为关机状态
        """

        mask = [False] * len(self.running_physical_machines)

        for i, pm in enumerate(self.running_physical_machines):
            mask[i] = (len(pm.running_virtual_nodes) > 0)

        self.running_physical_machines = np.array(self.running_physical_machines)[mask].tolist()

    def cost_calculation(self, t):
        """
        计算总能耗
        ========

        :param t: index of day
        :return:
        """
        for pm in self.running_physical_machines:
            self.cost += pm.daily_cost
        debug(f'{t}-th day\'s cost = {self.cost}')


def read():
    # 可以采购的服务器类型和数量
    # ======================

    # 第一行包含一个整数 N(1≤N≤100)，表示可以采购的服务器类型数量。
    N = int(input())
    # 接下来 N 行，每行描述一种类型的服务器，数据格式为：(型号, CPU 核数, 内存 大小, 硬件成本, 每日能耗成本)。
    # 例如(NV603, 92, 324, 53800, 500)表示一种服务器类型，其型号为 NV603，包含 92 个 CPU 核心，324G 内存，
    # 硬件成本为 53800，每日能耗成本为 500。CPU 核数，内存大小，硬件成本，每日能耗成本均为正整数。
    # 每台服务器的 CPU 核数以及内存大小不超过 1024，硬件成本不超过 5 × 105 ，每日能耗成本不超过 5000。
    # 服务器型号长度不超过 20，仅由数字和大小写英文字符构成。
    for n in range(N):
        line = re.sub('[ ()]', '', input().strip())
        if DEBUG: debug(line)

        model, cpu, ram, fixed_cost, daily_cost = line.split(',')
        possible_physical_machines[model] = {'cpu': int(cpu),
                                             'ram': int(ram),
                                             'fixed_cost': float(fixed_cost),
                                             'daily_cost': float(daily_cost)}
        # PhysicalMachine(model, int(cpu), int(ram), float(fixed_cost), float(daily_cost))
        # ==> end of N

    debug('====>')
    debug(f'Possible Physical Machines: {possible_physical_machines}')
    debug('')

    # 可供售卖的虚拟机类型和数量
    # ======================

    # 接下来一行包含一个整数 M(1≤M≤1000)，表示售卖的虚拟机类型数量。
    M = int(input())
    # 接下来 M 行，每行描述一种类型的虚拟机，数据格式为：(型号, CPU 核数, 内存大小, 是否双节点部署)。
    # 是否双节点部署用 0 和 1 表示，0 表示单节点部署，1 表示双节点部署。
    #   例如：
    #     (s3.small.1, 1, 1, 0) 表示一种虚拟机类型，其型号为 s3.small.1，所需 CPU 核数为 1，所需内存为 1G，单节点部署；
    #     (c3.8xlarge.2, 32, 64, 1) 表示一种虚拟机类型，其型号为 c3.8xlarge.2，所需 CPU 核数为 32，所需内存为 64G，双节点部署。
    # CPU 核数，内存大小均为正整数。
    # 对于每种类型的虚拟机，数据集保证至少存在一种服务器可以容纳。虚拟机型号长度不超过 20，仅由数字，大小写英文字符和'.'构成。
    for m in range(M):
        line = re.sub('[ ()]', '', input().strip())
        if DEBUG: debug(line)

        model, cpu, ram, double_type = line.split(',')
        possible_virtual_machines[model] = {'cpu': int(cpu),
                                            'ram': int(ram),
                                            'double_type': bool(double_type)}
        # VirtualMachine(model, int(cpu), int(ram), bool(double_type))
        # ==> end of M

    debug('====>')
    debug(f'Possible Virtual Machines: {possible_virtual_machines}')
    debug('')

    # 每日操作
    # ======

    monitor = Monitor()

    # 接下来一行包含一个整数 T(1≤T≤1000)，表示题目共会给出 T 天的用户请求序列数据。
    T = int(input())

    debug('====>')
    print(f'Daily operations...')
    debug('')

    # 接下来会按顺序给出 T 天的用户请求序列
    for t in range(T):
        # 对于每一天的数据，第一行包含一个非负整数 R 表示当天共有 R 条请求。
        R = int(input())

        """
        1. 扩容阶段
        这里是预先知道了新的一天的所有请求，然后进行扩容和删除
        """

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
            line = re.sub('[ ()]', '', input().strip())
            if DEBUG: debug(line)

            info = line.split(',')

            if len(info) >= 3 and info[0] == 'add':
                model, idx = info[1:]
                # 有可能会物理机分配不出来资源，这时候需要购置更多的物理机，因此也包含cost计算和更新
                monitor.add_virtual_machine(model, idx)
            elif len(info) >= 2 and info[0] == 'del':
                idx = info[-1]
                monitor.del_virtual_machine(idx)
            else:
                pass
            # ==> end of the r-th request

        """
        2. 然后是裁判系统给出所有请求
        """

        pass

        """
        3. 存量虚拟机的迁移
        """

        # 在完成扩容后，在处理每一天的新请求之前，你还可以对当前存量虚拟机进行一次迁移，即把虚拟机从一台服务器迁移至另一台服务器。
        # 对于单节点部署的虚拟机，将其从一台服务器的 A 节点迁移至 B 节点(或反之)也是允许的。
        # 迁移的目的服务器和节点必须有足够的资源容纳所迁移的虚拟机。
        # 迁移的虚拟机总量不超过当前存量虚拟机数量的千分之五。
        # 即假设当前有 n 台存量虚拟机，每天你可以迁移的虚拟机总量不得超过 5n/1000 向下取整。
        monitor.migration_physical_machine()

        """
        4. 空闲物理机的关闭
        """

        # 裁判程序会将当前有负载(至少部署了一台虚拟机)的服务器视为开机状态，没有任何负载的服务器视为关机状态
        monitor.power_off_physical_machine()

        """
        5. 成本核算
        """

        # 计算新的运营费用
        # 总成本包含两部分：购买服务器的整体硬件成本以及服务器消耗的整体能耗成本。
        # 整体硬件成本即将选手输出的方案中所有购买的服务器的硬件成本相加。
        # 整体能耗成本的计算方式为：在处理完每一天的所有操作后(包括迁移，创建和删除)
        monitor.cost_calculation(t)
        # ==> end of the t-th day


def main():
    # to read standard input
    # process
    # to write standard output
    # sys.stdout.flush()
    read()


if __name__ == "__main__":
    # 可供扩容的物理机
    possible_physical_machines = {}
    # 可供请求的虚拟机
    possible_virtual_machines = {}

    main()
