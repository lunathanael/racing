#include "game/maps.h"

#include <format>
#include <fstream>

void Maps::read_metadata(const std::string &index_file)
{
    std::ifstream file(index_file);
    assert(file.is_open());

    std::string line;
    while (std::getline(file, line))
    {
        size_t sep = line.find('|');
        if (sep == std::string::npos)
            continue;
        std::string file_stem = line.substr(0, sep);
        std::string display_name = line.substr(sep + 1);
        track_metadata.emplace_back(std::move(display_name), std::move(file_stem));
    }
}

void Maps::load_track()
{
    const std::string path = std::format("{}/{}.txt", data_dir, track_metadata[current_track_idx].file);
    current_track.load_from_file(path);
}

void Maps::previous_track()
{
    if (current_track_idx == 0)
        current_track_idx = size();
    --current_track_idx;
    load_track();
}

void Maps::next_track()
{
    if (++current_track_idx == size())
        current_track_idx = 0;
    load_track();
}

void Maps::set_track(const size_t &idx)
{
    assert(idx < size());
    current_track_idx = idx;
    load_track();
}

Maps::Maps(const std::string &index_file, const std::string &data_dir) : current_track_idx{}, data_dir{ data_dir }
{
    read_metadata(index_file);
    load_track();
}