#pragma once

#include "kmer_t.hpp"
#include <upcxx/upcxx.hpp>


struct HashMap {
    std::vector<upcxx::global_ptr<kmer_pair>> data;
    std::vector<upcxx::global_ptr<int>> used;
    const std::vector<upcxx::atomic_op> atomic_ops = {upcxx::atomic_op::compare_exchange};
    upcxx::atomic_domain<int>* ad = new upcxx::atomic_domain<int>(atomic_ops);
    size_t my_size;

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
    upcxx::global_ptr<kmer_pair> local_data = upcxx::new_array<kmer_pair>(size / upcxx::rank_n());
    upcxx::global_ptr<int> local_used = upcxx::new_array<int>(size / upcxx::rank_n());
    for (int i = 0; i < upcxx::rank_n(); i++) {
        data.push_back(upcxx::broadcast(local_data, i).wait());
        used.push_back(upcxx::broadcast(local_used, i).wait());
    }
    slots_per_node = size / upcxx::rank_n();
}

HashMap::~HashMap() {
    delete ad;
}

bool HashMap::insert(const kmer_pair& kmer) {
    uint64_t hash = kmer.hash();
    uint64_t probe = 0;
    bool success = false;
    do {
        uint64_t slot = (hash + probe++) % size();
        success = request_slot(slot);
        if (success) {
            write_slot(slot, kmer);
        }
    } while (!success && probe < size());
    return success;
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

void HashMap::write_slot(uint64_t slot, const kmer_pair& kmer) { 
    upcxx::rput(kmer, convert_slot_to_data_address(slot)).wait();
}

kmer_pair HashMap::read_slot(uint64_t slot) {
    upcxx::future<kmer_pair> result = upcxx::rget(convert_slot_to_data_address(slot));
    result.wait();
    return result.result();
}

bool HashMap::request_slot(uint64_t slot) {
    upcxx::future<int> res = ad->compare_exchange(convert_slot_to_used_address(slot), false, true, std::memory_order_release);
    res.wait();
    return !res.result();
}

size_t HashMap::size() const noexcept { return my_size; }
