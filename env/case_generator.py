import random
import time


class CaseGenerator:
    '''
    FJSP instance generator
    '''
    def __init__(self, job_init, num_mas, opes_per_job_min, opes_per_job_max, nums_ope=None, path='../data/',
                 flag_same_opes=True, flag_doc=False, is_fjsp=True):
        if nums_ope is None:
            nums_ope = []
        self.is_fjsp = is_fjsp
        self.flag_doc = flag_doc  # Whether save the instance to a file
        self.flag_same_opes = flag_same_opes
        self.nums_ope = nums_ope
        self.path = path  # Instance save path (relative path)
        self.job_init = job_init
        self.num_mas = num_mas

        self.mas_per_ope_min = 1
        self.mas_per_ope_max = num_mas if self.is_fjsp else 1
        self.opes_per_job_min = opes_per_job_min
        self.opes_per_job_max = opes_per_job_max
        self.proctime_per_ope_min = 1
        self.proctime_per_ope_max = 20 if self.is_fjsp else 99
        self.proctime_dev = 0.2

    def get_case(self, idx=0):
        '''
        Generate JSSP instance
        :param idx: The instance number
        '''
        self.num_jobs = self.job_init
        if self.is_fjsp:
            if not self.flag_same_opes:
                self.nums_ope = [random.randint(self.opes_per_job_min, self.opes_per_job_max) for _ in
                                 range(self.num_jobs)]
            self.num_opes = sum(self.nums_ope)
        else:
            # JSSP does not need flexibility in number of operations per job, so remove flexibility
            self.nums_ope = [self.num_mas] * self.num_jobs  # Same number of operations per job
        self.num_opes = sum(self.nums_ope)
        self.nums_option = [random.randint(self.mas_per_ope_min, self.mas_per_ope_max) for _ in range(self.num_opes)]
        self.num_options = sum(self.nums_option)
        self.ope_ma = []

        if self.is_fjsp:
            for val in self.nums_option:
                self.ope_ma = self.ope_ma + sorted(random.sample(range(1, self.num_mas + 1), val))
        else:
            for i in range(self.num_jobs):
                machines = list(range(1, self.num_mas + 1))
                for j in range(self.num_mas):
                    chosen_machine = random.sample(machines, 1)
                    self.ope_ma.append(chosen_machine[0])
                    machines.remove(chosen_machine[0])

        if self.is_fjsp:
            self.proc_time = []
            self.proc_times_mean = [random.randint(self.proctime_per_ope_min, self.proctime_per_ope_max) for _ in
                                    range(self.num_opes)]
            for i in range(len(self.nums_option)):
                low_bound = max(self.proctime_per_ope_min, round(self.proc_times_mean[i] * (1 - self.proctime_dev)))
                high_bound = min(self.proctime_per_ope_max, round(self.proc_times_mean[i] * (1 + self.proctime_dev)))
                proc_time_ope = [random.randint(low_bound, high_bound) for _ in range(self.nums_option[i])]
                self.proc_time = self.proc_time + proc_time_ope
        else:
            # Each operation now must be assigned to a specific machine
            self.machine_assignment = []
            for i in range(self.num_jobs):
                # Shuffle the list of available machines and select as many as needed for the operations
                available_machines = list(range(1, self.num_mas + 1))
                random.shuffle(available_machines)

                # Ensure each operation gets a unique machine
                self.machine_assignment.append(available_machines[:self.nums_ope[i]])

            # Generate processing times for each operation
            self.proc_time = []
            self.proc_times_mean = [random.randint(self.proctime_per_ope_min, self.proctime_per_ope_max) for _ in range(self.num_opes)]
            for i in range(len(self.nums_ope)):
                proc_time_ope = []
                for _ in range(self.nums_ope[i]):
                    time = random.randint(self.proctime_per_ope_min, self.proctime_per_ope_max)
                    proc_time_ope.append(time)

                self.proc_time = self.proc_time + proc_time_ope

        self.num_ope_biass = [sum(self.nums_ope[0:i]) for i in range(self.num_jobs)]
        self.num_ma_biass = [sum(self.nums_option[0:i]) for i in range(self.num_opes)]
        line0 = '{0}\t{1}\t{2}\n'.format(self.num_jobs, self.num_mas, self.num_options / self.num_opes)
        lines = []
        lines_doc = []
        lines.append(line0)
        lines_doc.append('{0}\t{1}\t{2}'.format(self.num_jobs, self.num_mas, self.num_options / self.num_opes))
        for i in range(self.num_jobs):
            flag = 0
            flag_time = 0
            flag_new_ope = 1
            idx_ope = -1
            idx_ma = 0
            line = []
            option_max = sum(self.nums_option[self.num_ope_biass[i]:(self.num_ope_biass[i]+self.nums_ope[i])])
            idx_option = 0
            while True:
                if flag == 0:
                    line.append(self.nums_ope[i])
                    flag += 1
                elif flag == flag_new_ope:
                    idx_ope += 1
                    idx_ma = 0
                    flag_new_ope += self.nums_option[self.num_ope_biass[i]+idx_ope] * 2 + 1
                    line.append(self.nums_option[self.num_ope_biass[i]+idx_ope])
                    flag += 1
                elif flag_time == 0:
                    line.append(self.ope_ma[self.num_ma_biass[self.num_ope_biass[i]+idx_ope] + idx_ma])
                    flag += 1
                    flag_time = 1
                else:
                    line.append(self.proc_time[self.num_ma_biass[self.num_ope_biass[i]+idx_ope] + idx_ma])
                    flag += 1
                    flag_time = 0
                    idx_option += 1
                    idx_ma += 1
                if idx_option == option_max:
                    str_line = " ".join([str(val) for val in line])
                    lines.append(str_line + '\n')
                    lines_doc.append(str_line)
                    break

        lines.append('\n')

        # Save the instance to a file if the flag is set
        if self.flag_doc:
            doc = open(self.path + '{0}j_{1}m_{2}.fjs'.format(self.num_jobs, self.num_mas, str.zfill(str(idx+1),3)),'a')
            for i in range(len(lines_doc)):
                print(lines_doc[i], file=doc)
            doc.close()
        return lines, self.num_jobs, self.num_jobs
