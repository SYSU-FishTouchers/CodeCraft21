import numpy as np

import os
import sys
import re

DEBUG = True


def debug(info: str, linesep=os.linesep):
    info += linesep
    sys.stderr.write(info)
    sys.stderr.flush()


def react(info: str, linesep=os.linesep):
    info += linesep
    sys.stdout.write(info)
    sys.stdout.flush()


class PhysicalMachine:
    def __init__(self, model, cpu, memory, fixed_cost, daily_cost):
        """
        :param model: 型号
        :param cpu: CPU 核数
        :param memory: 内存 大小
        :param fixed_cost: 硬件成本
        :param daily_cost: 每日能耗成本
        """

        self.model = model
        self.cpu = cpu
        self.memory = memory
        self.fixed_cost = fixed_cost
        self.daily_cost = daily_cost


class VirtualMachine:
    def __init__(self, model, cpu, memory, deploy_type):
        """
        :param model: 型号
        :param cpu: CPU 核数
        :param memory: 内存 大小
        :param deploy_type: 是否双节点部署
        """

        self.model = model
        self.cpu = cpu
        self.memory = memory
        # 直接表明部署节点数
        self.deploy_nodes = deploy_type + 1


def cost():
    """
    计算总能耗
    ========


    :return:
    """

    pass


def add_virtual_machine(model, idx):
    """
    用户新增虚拟机请求
    ===============

    :param model: 新增虚拟机的类型
    :param idx: 赋予新增虚拟机的id
    :return:
    """

    # 当物理机资源不足时，购置新的物理机
    cost()

    pass


def del_virtual_machine(idx):
    """
    用户删除虚拟机请求
    ===============

    :param idx: 请求删除虚拟机的id
    :return:
    """
    pass


def purchase_physical_machine(model, q):
    """
    购买新的物理机用于扩容
    ==================

    :param model: 需求的物理机类型
    :param q: 购买数量
    :return:
    """
    if DEBUG: N = 10
    assert 0 <= q <= N

    react(f'({model}, {q})')


def migration_physical_machine():
    """
    虚拟机迁移，核心算法
    =================

    在处理完每一天的所有操作后(包括迁移，创建和删除)，
    裁判程序会将当前有负载(至少部署了一台虚拟机)的服务器视为开机状态，
    没有任何负载的服务器视为关机状态

    :return:
    """


def power_down_physical_machine():
    """
    对没有负载的服务器关机
    ==================

    在处理完每一天的所有操作后(包括迁移，创建和删除)，
    裁判程序会将当前有负载(至少部署了一台虚拟机)的服务器视为开机状态，
    没有任何负载的服务器视为关机状态
    :return:
    """

    pass


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

        model, cpu, memory, fixed_cost, daily_cost = line.split(',')
        pm = PhysicalMachine(model, cpu, memory, fixed_cost, daily_cost)

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

        model, cpu, memory, deploy_type = line.split(',')
        vm = VirtualMachine(model, cpu, memory, deploy_type)

    # 每日操作
    # ======

    # 接下来一行包含一个整数 T(1≤T≤1000)，表示题目共会给出 T 天的用户请求序列数据。
    T = int(input())
    # 接下来会按顺序给出 T 天的用户请求序列
    for t in range(T):
        # 计算新的运营费用
        cost()

        # 对于每一天的数据，第一行包含一个非负整数 R 表示当天共有 R 条请求。
        R = int(input())
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
                add_virtual_machine(model, idx)
            elif len(info) >= 2 and info[0] == 'del':
                idx = info[1:]
                del_virtual_machine(idx)
            else:
                pass


def main():
    # to read standard input
    # process
    # to write standard output
    # sys.stdout.flush()
    read()


if __name__ == "__main__":
    main()
