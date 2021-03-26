from operator import lt

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
        self.volume = volume
        self.free = volume


class Numa:
    def __init__(self, cpu, ram, name):
        """
        Numa 节点

        :param cpu:
        :param ram:
        """
        self.cpu = Device(cpu)
        self.ram = Device(ram)
        self.name = name

    def free(self):
        return self.cpu.free, self.ram.free

    def volume(self):
        return self.cpu.volume, self.ram.volume

    def try_allocate(self, cpu, ram):
        if self.cpu.free >= cpu and self.ram.free >= ram:
            # 分配后，空余容量要减少
            self.cpu.free -= cpu
            self.ram.free -= ram
            return True
        return False

    def release(self, cpu, ram):
        self.cpu.free += cpu
        self.ram.free += ram


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
        self.A = Numa(cpu // 2, ram // 2, 'A')
        self.B = Numa(cpu // 2, ram // 2, 'B')

        # 服务器成本：数据中心使用每台服务器的成本由两部分构成：硬件成本和能耗成本。
        # 硬件成本是在采购服务器时的一次性支出，能耗成本是后续服务器使用过程中的持续支出。
        # 为了便于计算，我们以天为单位计算每台服务器的能耗成本。
        # 若一台服务器在一天内处于关机状态，则其不需要消耗任何能耗成本，否则我们需要支出其对应的能耗成本。
        self.fixed_cost = fixed_cost
        self.daily_cost = daily_cost

        # 目前正在此物理机上运行的虚拟机节点
        self.running_virtual_machines = {}

    def try_add_virtual_machines(self, vm: VirtualMachine, idx: int):
        """
        尝试指派一个虚拟机节点

        :param vm: 虚拟机
        :param idx: 虚拟机id
        :return: 双节点部署：'AB' ， 单节点部署：'A' 或 'B'
        """

        # 双节点部署指的是一台虚拟机所需的资源（CPU 和内存）必须由一台服务器的两个节点同时提供，
        # 并且每个节点提供总需求资源的一半。
        # 双节点部署的虚拟机保证其 CPU 和内存需求量都是偶数。
        if vm.double_type:
            assert vm.cpu % 2 == 0 and vm.ram % 2 == 0, f'Failed with {vm.cpu}C, {vm.ram}G, {"Double" if vm.double_type else "Single"}'
            cpu, ram = vm.cpu // 2, vm.ram // 2

            if self.A.try_allocate(cpu, ram) and self.B.try_allocate(cpu, ram):
                self.running_virtual_machines[idx] = (vm, 'AB')
                return 'AB'

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

            # 总是 Numa A 的剩余资源比 Numa B 少
            # if Numa A not less then Numa B
            # if not lt(self.A.free(), self.B.free()):

            # d1
            # 先放大的 1279887926
            # 先放小的  892109071

            # d2
            # ===================================================
            # 先放小的
            #
            # [Debug] Numa A:  105288 C  |  66853 G
            # [Debug] Numa B:  199923 C  |  71308 G
            #
            # [Debug]  999-th day's cost = 765588815
            # [Debug]          Time cost = 11.666s
            # [Debug]   Running machines : pm = 1877   vm = 20043
            #
            # [Debug] cost time: 11.66567087173462
            # ===================================================
            # 先放大的
            #
            # [Debug] Numa A:  50382 C  |  23642 G
            # [Debug] Numa B:  24312 C  |  18850 G
            #
            # [Debug]  999-th day's cost = 1030633981
            # [Debug]          Time cost = 15.943s
            # [Debug]   Running machines : pm = 2655   vm = 20043
            #
            # [Debug] cost time: 15.943521976470947
            # ===================================================
            # 不调整
            #
            # [Debug] Numa A:  110924 C  |  73241 G
            # [Debug] Numa B:  142892 C  |  78139 G
            #
            # [Debug]  999-th day's cost = 770409879
            # [Debug]          Time cost = 9.391s
            # [Debug]   Running machines : pm = 1880   vm = 20043
            #
            # [Debug] cost time: 9.391070127487183

            # if (self.A.free()[0] < self.B.free()[0]) and self.A.free()[1] < self.B.free()[1]:
            #     self.A, self.B = self.B, self.A

            if self.A.try_allocate(cpu, ram):
                self.running_virtual_machines[idx] = (vm, self.A.name)
                return self.A.name
            elif self.B.try_allocate(cpu, ram):
                self.running_virtual_machines[idx] = (vm, self.B.name)
                return self.B.name

        return ''

    def del_virtual_machines(self, idx):
        """
        删除给定下标的虚拟机节点

        :param idx:
        :return:
        """

        if idx not in self.running_virtual_machines.keys():
            return False

        vm, numa = self.running_virtual_machines[idx]

        if numa == self.A.name:
            self.A.release(vm.cpu, vm.ram)
        elif numa == self.B.name:
            self.B.release(vm.cpu, vm.ram)
        else:
            self.A.release(vm.cpu // 2, vm.ram // 2)
            self.B.release(vm.cpu // 2, vm.ram // 2)

        assert idx in self.running_virtual_machines.keys()
        self.running_virtual_machines.pop(idx)
        return True

    def get_free_resources(self):
        """
        获取物理机的占用情况

        :return: Numa A 的占用情况，Numa B 的占用情况
        """

        return self.A.free(), self.B.free()

    def get_virtual_machines(self):
        """
        获取所有正在运行的虚拟机节点

        :return: 所有正在运行的虚拟机节点
        """

        return self.running_virtual_machines
