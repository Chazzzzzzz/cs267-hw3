#pragma once

#include "kmer_t.hpp"
#include <upcxx/upcxx.hpp>

#define S 2000

struct HashMap {
    std::vector<upcxx::global_ptr<kmer_pair>> data;
    std::vector<upcxx::global_ptr<int>> used;
    const std::vector<upcxx::atomic_op> atomic_ops = {upcxx::atomic_op::compare_exchange, upcxx::atomic_op::fetch_add};
    upcxx::atomic_domain<int>* ad = new upcxx::atomic_domain<int>(atomic_ops);
    size_t my_size;
    kmer_pair local_cache[upcxx::rank_n()][S];
    int local_cache_pointer[upcxx::rank_n()] = {0};
    std::vector<upcxx::global_ptr<kmer_pair>> stack;
    std::vector<upcxx::global_ptr<int>> stack_pointer;

    size_t size() const noexcept;

    HashMap(size_t size);
    ~HashMap();


    // Most important functions: insert and retrieve
    // k-mers from the hash table.
    bool insert(const kmer_pair& kmer);
    bool find(const pkmer_t& key_kmer, kmer_pair& val_kmer);

    // Helper functions
    upcxx::global_ptr<int> convert_slot_to_used_address(uint64_t slot);
    upcxx::global_ptr<kmer_pair> convert_slot_to_data_address(uint64_t slot);

    // Write and read to a logical data slot in the table.
    void write_slot(uint64_t slot, const kmer_pair& kmer);
    kmer_pair read_slot(uint64_t slot);

    // Request a slot or check if it's already used.
    bool request_slot(uint64_t slot);
    bool slot_used(uint64_t slot);
};

static int slots_per_node;

HashMap::HashMap(size_t size) {
    size = (size + upcxx::rank_n() - 1) / upcxx::rank_n() * upcxx::rank_n();
    my_size = size;
    upcxx::global_ptr<kmer_pair> local_stack = upcxx::new_array<kmer_pair>(size / upcxx::rank_n());
    upcxx::global_ptr<int> local_stack_pointer = 0;
    upcxx::global_ptr<kmer_pair> local_data = upcxx::new_array<kmer_pair>(size / upcxx::rank_n());
    upcxx::global_ptr<int> local_used = upcxx::new_array<int>(size / upcxx::rank_n());
    for (int i = 0; i < upcxx::rank_n(); i++) {
        stack.push_back(upcxx::broadcast(local_stack, i).wait());
        data.push_back(upcxx::broadcast(local_data, i).wait());
        used.push_back(upcxx::broadcast(local_used, i).wait());
        stack_pointer.push_back(upcxx::broadcast(local_stack_pointer, i).wait());
    }
    slots_per_node = size / upcxx::rank_n();
}

HashMap::~HashMap() {
    delete ad;
}

bool HashMap::insert(const kmer_pair& kmer, bool end) {
    uint64_t hash = kmer.hash();
    uint64_t probe = 0;
    bool success = false;
    do {
        uint64_t slot = (hash + probe++) % size();
        int node = slot / slots_per_node;
        if(node == upcxx::rank_me()) {
            success = request_local_slot(slot);
            if(success) {
                write_local_slot(slot, kmer);
            }
        } else {
            if(!end) {
                write_to_local_cache(slot, kmer);
                success = true;
            } else {
                success = request_slot(slot);
                if(success) {
                    write_slot(slot, kmer);
                }
            }
        }
    } while (!success && probe < size());
    return success;
}

bool HashMap::finish_insert() {
    clear_cache();
    stack_to_hashtable();
}

bool HashMap::clear_cache() {
    for(int i = 0; i < upcxx::rank_n(); i++) {
        write_to_stack(i, local_cache[i], local_cache_pointer[i]);
    }
}

bool HashMap::stack_to_hashtable() {
    for(int i = 0; i < local_stack_pointer; i++) {
        insert(local_stack[i], true);
    }
}

bool HashMap::find(const pkmer_t& key_kmer, kmer_pair& val_kmer) {
    uint64_t hash = key_kmer.hash();
    uint64_t probe = 0;
    bool success = false;
    do {
        uint64_t slot = (hash + probe++) % size();
        val_kmer = read_slot(slot);
        if (val_kmer.kmer == key_kmer) {
            success = true;
        }
    } while (!success && probe < size());
    return success;
}

upcxx::global_ptr<int> HashMap::convert_slot_to_used_address(uint64_t slot) {
    int node = slot / slots_per_node;
    int offset = slot % slots_per_node;
    return used[node] + offset;
}

upcxx::global_ptr<kmer_pair> HashMap::convert_slot_to_data_address(uint64_t slot) {
    int node = slot / slots_per_node;
    int offset = slot % slots_per_node;
    return data[node] + offset;
}

bool HashMap::write_to_local_cache(uint64_t slot, const kmer_pair& kmer) {
    int node = slot / slots_per_node;
    local_cache[node][local_cache_pointer[node]++] = kmer;
    if(local_cache_pointer[node] == S) {
        write_to_stack(node, local_cache[node], S);
        local_cache_pointer[node] = 0;
    }

}

void HashMap::write_to_stack(int node, kmer_pair local_cache[], int length) {
    upcxx::future<int> res = ad->fetch_add(stack_pointer[node], length, std::memory_order_release);
    res.wait();
    upcxx::rput(local_cache, stack[node] + res.result() - length, length).wait();
}

void HashMap::write_slot(uint64_t slot, const kmer_pair& kmer) { 
    upcxx::rput(kmer, convert_slot_to_data_address(slot)).wait();
}

void HashMap::write_local_slot(uint64_t slot, const kmer_pair& kmer) {
    int offset = slot % slots_per_node;
    local_data[offset] = kmer;
}

kmer_pair HashMap::read_slot(uint64_t slot) {
    upcxx::future<kmer_pair> result = upcxx::rget(convert_slot_to_data_address(slot));
    result.wait();
    return result.result();
}

bool HashMap::request_local_slot(uint64_t slot) {
    int offset = slot % slots_per_node;
    if(local_used[offset] != 0) {
        return false;
    } else {
        local_used[offset] = 1;
        return true;
    }
}

bool HashMap::request_slot(uint64_t slot) {
    upcxx::future<int> res = ad->compare_exchange(convert_slot_to_used_address(slot), false, true, std::memory_order_release);
    res.wait();
    return !res.result();
}

size_t HashMap::size() const noexcept { return my_size; }
