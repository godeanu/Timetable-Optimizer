import numpy as np
from utils import *
from check_constraints import *
from copy import deepcopy
import time
import random
from math import sqrt, log


def parse_time_constraint(constraint):
    if constraint.startswith('!') and '-' in constraint[1:]:
        try:
            parts = constraint[1:].split('-')
            return (int(parts[0]), int(parts[1]))
        except ValueError:
            return None
    return None


class TimeTableState:
    def __init__(self, timetable, timetable_specs, subject_students, initial, file, subject_room_availability):
            self.timetable_specs = timetable_specs
            self.file = file
            if initial==0:
                self.initialize()
                self.subject_room_availability = self.calculate_room_availability()
            else:
                self.timetable = timetable
                self.subject_students = {k: v for k, v in subject_students.items()}
                self.subject_room_availability = subject_room_availability
            self.nrconflicts = TimeTableState.compute_score(self)
    
    def initialize(self):
        self.timetable = self.create_empty_timetable()
        self.subject_students = {}
        for subject, count in self.timetable_specs['Materii'].items():
            current_count = self.subject_students.get(subject, 0)
            self.subject_students[subject] = current_count + count

    def create_empty_timetable(self):
        return {
            day: {
                eval(interval): {room: None for room in self.timetable_specs['Sali']}
                for interval in self.timetable_specs['Intervale']
            }
            for day in self.timetable_specs['Zile']
        }
    

    def calculate_teacher_breaks(self, teacher, day):
        intervals = []
        for interval in self.timetable[day]:
            for room in self.timetable[day][interval]:
                if self.timetable[day][interval][room] and self.timetable[day][interval][room][0] == teacher:
                    intervals.append(interval)

        if len(intervals) < 2:
            return 0  
        
        intervals = sorted(intervals, key=lambda x: x[0])

        breaks = 0
        for i in range(len(intervals) - 1):
            end_of_current_class = intervals[i][1]
            start_of_next_class = intervals[i + 1][0]
            break_duration = start_of_next_class - end_of_current_class
            breaks += max(0, break_duration) 
        return breaks/2
    

    def check_teacher_break_violations(self, teacher):
        max_allowed_break = self.extract_max_break(teacher)
        for day in self.timetable:
            breaks = self.calculate_teacher_breaks(teacher, day)
            if breaks > max_allowed_break:
                return True
        return False

    def extract_max_break(self, teacher):
        for constraint in self.timetable_specs['Profesori'][teacher]['Constrangeri']:
            if constraint.startswith('!Pauza >'):
                return int(constraint.split('>')[1].strip())
        return float('inf')
    
    def calculate_room_availability(self):
        '''Checking how many rooms can a subject be assigned to - 
        used for making sure we add the subject with least rooms to those ones first'''
        subject_room_count = {subject: 0 for subject in self.timetable_specs['Materii']}
        for room, details in self.timetable_specs['Sali'].items():
            for subject in details['Materii']:
                subject_room_count[subject] += 1
        return subject_room_count


    def compute_score(self):
        conflicts = 0
        if self.file == 'orar_bonus_exact':
            assignment_percentage = self.calculate_assignment_percentage()
        else:
            assignment_percentage = 0
        for day in self.timetable:
            conflicts += self.conflicts_per_day(day, self.subject_room_availability)
            if assignment_percentage > 66: 
                for teacher in self.timetable_specs['Profesori']:
                    if self.check_teacher_break_violations(teacher):
                        conflicts += 0.5
        return conflicts
    
    def calculate_assignment_percentage(self):
        total_slots = sum(len(self.timetable[day][interval]) for day in self.timetable for interval in self.timetable[day])
        filled_slots = sum(1 for day in self.timetable for interval in self.timetable[day] for room in self.timetable[day][interval] if self.timetable[day][interval][room] is not None)
        return (filled_slots / total_slots) * 100

    def conflicts_per_day(self, day, subject_room_availability):
        day_conflicts = 0
        for interval in self.timetable[day]:
            day_conflicts += self.conflicts_per_interval(day, interval, subject_room_availability)
        return day_conflicts

    def conflicts_per_interval(self, day, interval, subject_room_availability):
        interval_conflicts = 0

        for room in self.timetable[day][interval]:
            if self.timetable[day][interval][room]:
                teacher = self.timetable[day][interval][room][0]
                subject = self.timetable[day][interval][room][1]
                if subject_room_availability[subject] == 1:
                    interval_conflicts -= 3
                room_subjects = self.timetable_specs['Sali'][room]['Materii']


                interval_conflicts += self.check_professor_conflicts(teacher, day, interval)
                if subject in room_subjects:
                    if len(room_subjects) == 1:
                        interval_conflicts -= 7
                    elif len(room_subjects) > 1:
                        interval_conflicts += 0.5
        return interval_conflicts

    def check_professor_conflicts(self, professor, day, interval):
        conflicts = 0
        current_interval_start, current_interval_end = interval

        if f'!{day}' in self.timetable_specs['Profesori'][professor]['Constrangeri']:
            conflicts += 1 

        for constraint in self.timetable_specs['Profesori'][professor]['Constrangeri']:
            time_interval = parse_time_constraint(constraint)
            if time_interval:
                constraint_start, constraint_end = time_interval
                if constraint_start <= current_interval_start and constraint_end >= current_interval_end:
                    conflicts += 1

        return conflicts
    
    def is_final(self):
        return sum(self.subject_students.values()) == 0
                                  
    def get_neighbours(self, timetable_specs):
        neighbours = []

        for day in self.timetable:
            for interval in self.timetable[day]:
                for room in timetable_specs['Sali']:
                    if self.timetable[day][interval].get(room) is None:
                        for subject in self.timetable_specs['Materii']:
                            if self.subject_students[subject] > 0 and subject in timetable_specs['Sali'][room]['Materii']:
                                for teacher in timetable_specs['Profesori']:
                                    if subject in timetable_specs['Profesori'][teacher]['Materii']:
                                        if self.can_teach(teacher, day, interval):
                                            new_timetable = deepcopy(self.timetable)
                                            new_subject_students = {k: v for k, v in self.subject_students.items()}
                                            new_timetable[day][interval][room] = (teacher, subject)
                                            if new_subject_students[subject] > timetable_specs['Sali'][room]['Capacitate']:
                                                new_subject_students[subject] -= timetable_specs['Sali'][room]['Capacitate']
                                            else:
                                                new_subject_students[subject] = 0
                                            neighbours.append(TimeTableState(new_timetable, timetable_specs, new_subject_students, 1, self.file, self.subject_room_availability))
        return neighbours
    
    def can_teach(self, teacher, day, interval):
        teacher_hours = 0
        for d in self.timetable:
            for i in self.timetable[d]:
                for r in self.timetable[d][i]:
                    if self.timetable[d][i][r] and self.timetable[d][i][r][0] == teacher:
                        teacher_hours += 1
        if teacher_hours >= 7:
            return False
        if self.timetable[day][interval]:
            for room in self.timetable[day][interval]:
                if self.timetable[day][interval][room] and self.timetable[day][interval][room][0] == teacher:
                    return False
        return True
    
def heuristic(state):
    return 5*state.nrconflicts + sum([state.subject_students[subject] for subject in state.subject_students])

def hill_climbing(init, timetable_specs, max_iters=500):
    iters, states = 0, 0
    state = deepcopy(init)
    while iters < max_iters:
        iters += 1
        best = state
        neighbours = state.get_neighbours(timetable_specs)
        states += len(neighbours)
        for neighbor in neighbours:
            if heuristic(neighbor) < heuristic(best):
                best = neighbor
        if best == state:
            break
        state = best

    return iters, states, state

class Node:
    def __init__(self, state, parent, timetable_specs):
        self.timetable_specs = timetable_specs
        self.state = state
        self.parent = parent
        self.children = {}
        self.N = 0
        self.Q = 0 

    def expand(self):
        available_actions = self.state.get_neighbours(self.timetable_specs)
        for action in available_actions:
            self.children[action] = Node(action, self, self.timetable_specs)

        
    def softmax(x :np.array) -> float:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()


    def select(self):
        CP = 1.0/ sqrt(2.0)

        if not self.children:
            return self
        N_node = self.N
        best_action, best_score = None, -1000
        for action in self.children:
            child = self.children[action]
            if child.N == 0:
                score = 0
            else:
                score = child.Q / child.N + CP * sqrt(2 * log(N_node) / child.N)
            if score > best_score:
                best_action, best_score = action, score
        return self.children[best_action]
    
    def simulate(self):
        current_simulation_state = deepcopy(self.state)
        while not current_simulation_state.is_final():
            possible_moves = current_simulation_state.get_neighbours(self.timetable_specs)
            if not possible_moves:
                break
            action = np.random.choice(possible_moves, p=Node.softmax(np.array([-heuristic(state) for state in possible_moves if state is not None])))
            current_simulation_state = action
        return heuristic(current_simulation_state)     

    def backpropagate(self, reward):
        node = self
        while node:
            node.N += 1
            node.Q += reward
            node = node.parent     

    def best_child(self):
        if not self.children:
            return self
        best_child, best_score = None, 100
        for child in self.children:
            if self.children[child].N == 0:
                return self.children[child]
            score = self.children[child].Q / self.children[child].N
            if score < best_score:
                best_child, best_score = self.children[child], score
        return best_child
    
def mcts(root_state, timetable_specs, iterations):
    root_node = Node(root_state, None, timetable_specs)
    for _ in range(iterations):
        node = root_node
        while True:
            if node.state.is_final() or len(node.children) < len(node.state.get_neighbours(timetable_specs)):
                break
            else:
                node = node.select()

        if not node.children and not node.state.is_final():
            node.expand() 

        while node.children:
            node = random.choice(list(node.children.values()))

        reward = node.simulate()
        node.backpropagate(reward)

    return root_node.best_child().state
if __name__ == '__main__':

    start_time = time.time()
    if len(sys.argv) < 3: 
        print('Run like this:\npython3 orar.py dummy hc')
        sys.exit(0)
    file = sys.argv[1]
    algorithm = sys.argv[2]
    if algorithm == 'hc' or algorithm == 'mcts':
        pass
    else:
        print('The algorithm should only be mcts or hc')
        sys.exit(0)
    yaml_dict = read_yaml_file(f'inputs/{file}.yaml')
    
    if algorithm == 'hc':
        print('running...\n')
        timetable = {}
        state = TimeTableState(timetable, yaml_dict,  {}, 0, file, {})
        
        iters, states, best_timetable = hill_climbing(state, yaml_dict, 1000)
        print(best_timetable.timetable)
        print(pretty_print_timetable(best_timetable.timetable, f'inputs/{file}.yaml'))
        print()
        print(iters , 'iterations')
        print(states, 'states\n')
        print(check_optional_constraints(best_timetable.timetable, yaml_dict), 'optional constraints violated')
        print(check_mandatory_constraints(best_timetable.timetable, yaml_dict), 'mandatory constraints violated')

        output_path = f'outputs/{file}.txt'
        with open(output_path, 'w') as f:
            f.write(pretty_print_timetable(best_timetable.timetable, f'inputs/{file}.yaml'))

    elif algorithm == 'mcts':
        print('running...\n')
        timetable = {}
        table = TimeTableState(timetable, yaml_dict, {}, 0, file, {})
        for subject in yaml_dict['Materii']:
            while table.subject_students[subject] > 0:
                table = mcts(table, yaml_dict, 77)
        print(table.timetable)
        print(pretty_print_timetable(table.timetable, f'inputs/{file}.yaml'))
        print()
        print(check_mandatory_constraints(table.timetable, yaml_dict), 'mandatory constraints violated')
        print(check_optional_constraints(table.timetable, yaml_dict), 'optional constraints violated')
        output_path = f'outputs/{file}.txt'
        with open(output_path, 'w') as f:
            f.write(pretty_print_timetable(table.timetable, f'inputs/{file}.yaml'))

       
    print()
    end_time = time.time()
    print(f"Execution time: {end_time - start_time} seconds")
    

