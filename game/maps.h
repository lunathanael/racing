#pragma once

#include "game/track.h"

#include <string>
#include <vector>

struct TrackMetadata
{
    std::string name;
    std::string file;
};

class Maps
{
    Track current_track;
    std::vector<TrackMetadata> track_metadata;
    size_t current_track_idx;
    const std::string data_dir;

    void read_metadata(const std::string &index_file);
    void load_track();

public:
    Maps(const std::string &index_file, const std::string &data_dir);

    decltype(auto) size() const { return track_metadata.size(); }
    decltype(auto) get_current_idx() const { return (current_track_idx); }
    decltype(auto) get_current_track() const { return (current_track); }

    decltype(auto) get_metadata(const size_t &idx) const { return track_metadata[idx]; }

    void previous_track();
    void next_track();
    void set_track(const size_t &idx);
};