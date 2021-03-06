#pragma once

#include "kmer_t.hpp"
#include <upcxx/upcxx.hpp>

#define S 80000

struct HashMap {
    upcxx::global_ptr<kmer_pair> local_data;
    upcxx::global_ptr<int> local_used;
    std::vector<upcxx::global_ptr<kmer_pair>> data;
    std::vector<upcxx::global_ptr<int>> used;
    const std::vector<upcxx::atomic_op> atomic_ops = {upcxx::atomic_op::compare_exchange, upcxx::atomic_op::fetch_add};
    upcxx::atomic_domain<int>* ad = new upcxx::atomic_domain<int>(atomic_ops);
    size_t my_size;
    kmer_pair** local_cache;
    int *local_cache_pointer;
    upcxx::global_ptr<kmer_pair> local_stack;
    upcxx::global_ptr<int> local_stack_pointer;
    std::vector<upcxx::global_ptr<kmer_pair>> stack;
    std::vector<upcxx::global_ptr<int>> stack_pointer;

    size_t size() const noexcept;

    HashMap(size_t size);
    ~HashMap();


    // Most important functions: insert and retrieve
    // k-mers from the hash table.
    bool insert(kmer_pair& kmer, bool end);
    bool find(const pkmer_t& key_kmer, kmer_pair& val_kmer);

    // Helper functions
    upcxx::global_ptr<int> convert_slot_to_used_address(uint64_t slot);
    upcxx::global_ptr<kmer_pair> convert_slot_to_data_address(uint64_t slot);

    // Write and read to a logical data slot in the table.
    void write_slot(uint64_t slot, kmer_pair& kmer);
    void write_local_slot(uint64_t slot, kmer_pair& kmer);
    void write_to_stack(int node, kmer_pair local_cache[], int length);
    void write_to_local_cache(uint64_t slot, kmer_pair& kmer);
    kmer_pair read_slot(uint64_t slot);

    // Request a slot or check if it's already used.
    bool request_slot(uint64_t slot);
    bool request_local_slot(uint64_t slot);
    bool slot_used(uint64_t slot);
    
    void finish_insert();
};

static int slots_per_node;

HashMap::HashMap(size_t size) {
    if (upcxx::rank_n() > 1) {
        size = (size + upcxx::rank_n() - 1) / upcxx::rank_n() * upcxx::rank_n();
        my_size = size;
        local_cache = new kmer_pair*[upcxx::rank_n()];
        local_cache_pointer = new int[upcxx::rank_n()];
        for( int i = 0; i < upcxx::rank_n(); i++) {
            local_cache[i] = new kmer_pair[S];
            local_cache_pointer[i] = 0;
        }
        local_stack = upcxx::new_array<kmer_pair>(size / upcxx::rank_n());
        local_stack_pointer = upcxx::new_<int>(0);
        local_data = upcxx::new_array<kmer_pair>(size / upcxx::rank_n());
        local_used = upcxx::new_array<int>(size / upcxx::rank_n());
        for (int i = 0; i < upcxx::rank_n(); i++) {
            stack.push_back(upcxx::broadcast(local_stack, i).wait());
            data.push_back(upcxx::broadcast(local_data, i).wait());
            used.push_back(upcxx::broadcast(local_used, i).wait());
            stack_pointer.push_back(upcxx::broadcast(local_stack_pointer, i).wait());
        }
        slots_per_node = size / upcxx::rank_n();
    } else {
        size = (size + upcxx::rank_n() - 1) / upcxx::rank_n() * upcxx::rank_n();
        my_size = size;
        local_data = upcxx::new_array<kmer_pair>(size / upcxx::rank_n());
        local_used = upcxx::new_array<int>(size / upcxx::rank_n());
        for (int i = 0; i < upcxx::rank_n(); i++) {
            data.push_back(upcxx::broadcast(local_data, i).wait());
            used.push_back(upcxx::broadcast(local_used, i).wait());
        }
        slots_per_node = size / upcxx::rank_n();
    }
}

HashMap::~HashMap() {
    delete ad;
}

bool HashMap::insert(kmer_pair& kmer, bool end) {
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

void HashMap::finish_insert() {
    if(upcxx::rank_n() > 1){
        // clear cache
        for(int i = 0; i < upcxx::rank_n(); i++) {
            write_to_stack(i, local_cache[i], local_cache_pointer[i]);
        }
        upcxx::barrier();
        kmer_pair* local_stack_ptr = local_stack.local();
        // stack to hashtable
        for(int i = 0; i < *(local_stack_pointer.local()); i++) {
            insert(local_stack_ptr[i], true);
        }
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

void HashMap::write_to_local_cache(uint64_t slot, kmer_pair& kmer) {
    int node = slot / slots_per_node;
    local_cache[node][local_cache_pointer[node]++] = kmer;
    if(local_cache_pointer[node] == S) {
        write_to_stack(node, local_cache[node], S);
        local_cache_pointer[node] = 0;
    }
}

void HashMap::write_to_stack(int node, kmer_pair local_cache[], int length) {
    if(length == 0)
	return;
    upcxx::future<int> res = ad->fetch_add(stack_pointer[node], length, std::memory_order_release);
    res.wait();
    upcxx::rput(local_cache, stack[node] + res.result(), length).wait();
}

void HashMap::write_slot(uint64_t slot, kmer_pair& kmer) { 
    upcxx::rput(kmer, convert_slot_to_data_address(slot)).wait();
}

void HashMap::write_local_slot(uint64_t slot, kmer_pair& kmer) {
    int offset = slot % slots_per_node;
    kmer_pair* local_data_pointer = local_data.local();
    local_data_pointer[offset] = kmer;
}

kmer_pair HashMap::read_slot(uint64_t slot) {
    upcxx::future<kmer_pair> result = upcxx::rget(convert_slot_to_data_address(slot));
    result.wait();
    return result.result();
}

bool HashMap::request_local_slot(uint64_t slot) {
    int offset = slot % slots_per_node;
    int* local_used_pointer = local_used.local();
    if(local_used_pointer[offset] != 0) {
        return false;
    } else {
        local_used_pointer[offset] = 1;
        return true;
    }
}

bool HashMap::request_slot(uint64_t slot) {
    upcxx::future<int> res = ad->compare_exchange(convert_slot_to_used_address(slot), false, true, std::memory_order_release);
    res.wait();
    return !res.result();
}

size_t HashMap::size() const noexcept { return my_size; }
