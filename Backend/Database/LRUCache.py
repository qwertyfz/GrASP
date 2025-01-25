from typing import Dict
import numpy as np


class Node:
    def __init__(self, key):
        self.key = key
        self.prev = None
        self.next = None


class LRUCache:
    def __init__(self, capacity):
        self.capacity = capacity
        self.cache: Dict[str, Node] = {}
        self.head = Node(None)
        self.tail = Node(None)
        self.head.next = self.tail
        self.tail.prev = self.head
        self.hit_count = 0
        self.miss_count = 0
        self.total_pres = 0
        self.access_to_evicted_count = 0
        self.access_to_evicted_count_np = 0
        self.all_ever_inserted = set()
        self.evicted_keys : Dict[str, int] = {}
        self.evicted_counter_seq = []
        self.reinsertion_count = 0

    def get_all_content(self):
        return list(self.cache.keys())

    def get(self, key, increase_hit: bool):
        if key in self.cache:
            node = self.cache[key]
            self._remove(node)
            self._add(node)
            if increase_hit:
                self.hit_count += 1
            return node.key
        if increase_hit:
            self.miss_count += 1
        return None

    def put(self, key, increase_hit: bool, insert_type='n', np_contents=set(), check_insertions=False):
        if key not in self.cache:
            if increase_hit:
                self.miss_count += 1

            if insert_type == 'p':
                self.total_pres += 1

            if check_insertions:
                if key in self.all_ever_inserted:
                    self.reinsertion_count += 1
                    step_count = self.evicted_keys[key]
                    del self.evicted_keys[key]
                    
                    if increase_hit: # An evicted block is reaccessed
                        self.evicted_counter_seq.append(step_count)
                        self.access_to_evicted_count += 1
                        if key in np_contents:
                            self.access_to_evicted_count_np += 1
        else:
            if increase_hit:
                self.hit_count += 1
            self._remove(self.cache[key])
        node = Node(key)
        self._add(node)
        self.cache[key] = node
        if check_insertions: self.all_ever_inserted.add(key)
        if len(self.cache) > self.capacity:
            node_to_remove = self.head.next
            if check_insertions: self.evicted_keys[node_to_remove.key] = 0
            self._remove(node_to_remove)
            del self.cache[node_to_remove.key]

    def get_size(self):
        return len(self.cache)

    def increase_evicted_keys_counter(self):
        self.evicted_keys = {k: v + 1 for k, v in self.evicted_keys.items()}

    def clear(self):
        self.cache: Dict[str, Node] = {}
        self.head = Node(None)
        self.tail = Node(None)
        self.head.next = self.tail
        self.tail.prev = self.head
        self.hit_count = 0
        self.miss_count = 0
        self.total_pres = 0
        self.access_to_evicted_count = 0
        self.access_to_evicted_count_np = 0
        self.all_ever_inserted = set()
        self.evicted_keys : Dict[str, int] = {}
        self.evicted_counter_seq = []
        self.reinsertion_count = 0  
    

    def _add(self, node):
        prev_node = self.tail.prev
        prev_node.next = node
        node.prev = prev_node
        node.next = self.tail
        self.tail.prev = node

    def _remove(self, node):
        prev_node = node.prev
        next_node = node.next
        prev_node.next = next_node
        next_node.prev = prev_node

    def get_hit_ratio(self):
        if self.get_total_access() == 0:
            return 0
        return self.hit_count/self.get_total_access()

    def get_total_access(self):
        return self.hit_count + self.miss_count

    def __repr__(self):
        return f"LRUCache({self.capacity})"

    def __str__(self):
        return f"LRUCache({self.capacity}): {list(self.cache.keys())}"

    def report(self, name):
        return f'{name}: total access = {self.get_total_access()}, total prefetches = {self.total_pres} ' \
               f'hits = {self.hit_count}, hit ratio = {self.get_hit_ratio()}'

    def get_full_report(self, name):
        evicted_reaccess_counter_mean = 0
        evicted_reaccess_counter_std = 0
        if len(self.evicted_counter_seq):
            reaccess_counter_array = np.array(self.evicted_counter_seq)
            evicted_reaccess_counter_mean = np.mean(reaccess_counter_array)
            evicted_reaccess_counter_std = np.std(reaccess_counter_array)

        return f'{name}: total access = {self.get_total_access()}, total prefetches = {self.total_pres} ' \
               f'hits = {self.hit_count}, hit ratio = {self.get_hit_ratio()}'\
               f'\n\t\tevicted_reaccesses = {self.access_to_evicted_count}, evicted_reaccesses_np = {self.access_to_evicted_count_np}, '\
               f'reinsertions = {self.reinsertion_count}, '\
               f'evicted_reaccess_counter_mean = {evicted_reaccess_counter_mean}, evicted_reaccess_counter_std = {evicted_reaccess_counter_std}'


    def report_dict(self):
        res = {
            'total_access': self.get_total_access(),
            'total_prefetches': self.total_pres,
            'hits': self.hit_count,
            'hit_ratio': self.get_hit_ratio(),
            'final_cache_usage': len(self.cache),
            'evicted_reaccesses' : self.access_to_evicted_count,
            'evicted_reaccesses_np' : self.access_to_evicted_count_np,
            'reinsertions': self.reinsertion_count
        }

        if len(self.evicted_counter_seq):
            reaccess_counter_array = np.array(self.evicted_counter_seq)
            res['evicted_reaccess_counter_mean'] = np.mean(reaccess_counter_array)
            res['evicted_reaccess_counter_std'] = np.std(reaccess_counter_array)
        
        else:
            res['evicted_reaccess_counter_mean'] = 0
            res['evicted_reaccess_counter_std'] = 0

        return res
