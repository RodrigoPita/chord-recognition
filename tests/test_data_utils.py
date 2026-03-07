from pathlib import Path

import pytest
from pydantic import ValidationError

from chord_rec.data_utils import Song, get_files, name_from_path, songs_from_files, txt_to_csv, txt_to_csv_aligned


class TestSong:
    def test_valid_song(self, tmp_path):
        audio = tmp_path / "artist - song.wav"
        label = tmp_path / "song.csv"
        audio.touch()
        label.touch()

        song = Song(audio_path=audio, annotation_path=label, name="song")
        assert song.name == "song"
        assert song.audio_path == audio
        assert song.annotation_path == label

    def test_missing_audio_path_raises(self, tmp_path):
        label = tmp_path / "song.csv"
        label.touch()

        with pytest.raises(ValidationError):
            Song(audio_path=tmp_path / "nonexistent.wav", annotation_path=label, name="song")

    def test_missing_annotation_path_raises(self, tmp_path):
        audio = tmp_path / "song.wav"
        audio.touch()

        with pytest.raises(ValidationError):
            Song(audio_path=audio, annotation_path=tmp_path / "nonexistent.csv", name="song")


class TestNameFromPath:
    def test_artist_song_format(self):
        assert name_from_path(Path("Artist - Song Title.wav")) == "Song Title"

    def test_no_separator_falls_back_to_stem(self):
        assert name_from_path(Path("SongTitle.wav")) == "SongTitle"

    def test_only_first_separator_is_used(self):
        assert name_from_path(Path("A - B - C.wav")) == "B - C"

    def test_accepts_string(self):
        assert name_from_path("Artist - Song.wav") == "Song"


class TestSongsFromFiles:
    def test_builds_list_from_matching_files(self, tmp_path):
        audio1 = tmp_path / "Artist - Song One.wav"
        audio2 = tmp_path / "Artist - Song Two.wav"
        label1 = tmp_path / "song_one.csv"
        label2 = tmp_path / "song_two.csv"
        for f in [audio1, audio2, label1, label2]:
            f.touch()

        songs = songs_from_files([audio1, audio2], [label1, label2])

        assert len(songs) == 2
        assert songs[0].name == "Song One"
        assert songs[1].name == "Song Two"


class TestGetFiles:
    def test_reads_filenames_from_txt(self, tmp_path):
        audio1 = tmp_path / "song1.wav"
        audio2 = tmp_path / "song2.wav"
        audio1.touch()
        audio2.touch()

        index_file = tmp_path / "index.txt"
        index_file.write_text("song1.wav\nsong2.wav")

        result = get_files(index_file, tmp_path)
        assert result == [tmp_path / "song1.wav", tmp_path / "song2.wav"]

    def test_ignores_empty_lines(self, tmp_path):
        index_file = tmp_path / "index.txt"
        index_file.write_text("song.wav\n\n")
        (tmp_path / "song.wav").touch()

        result = get_files(index_file, tmp_path)
        assert len(result) == 1


class TestTxtToCsv:
    _sample_txt = "0.0\t2.5\tAm\n2.5\t5.0\tC\n5.0\t7.5\tG"

    def test_writes_csv_with_header(self, tmp_path):
        src = tmp_path / "ann.txt"
        dst = tmp_path / "ann.csv"
        src.write_text(self._sample_txt)

        txt_to_csv(src, dst)

        lines = dst.read_text().splitlines()
        assert lines[0] == '"Start";"End";"Label"'
        assert len(lines) == 4  # header + 3 data rows

    def test_csv_row_format(self, tmp_path):
        src = tmp_path / "ann.txt"
        dst = tmp_path / "ann.csv"
        src.write_text("0.0\t2.5\tAm")

        txt_to_csv(src, dst)

        lines = dst.read_text().splitlines()
        assert lines[1] == '0.0;2.5;"Am"'

    def test_skips_malformed_lines(self, tmp_path):
        src = tmp_path / "ann.txt"
        dst = tmp_path / "ann.csv"
        src.write_text("0.0\t2.5\tAm\nbad_line")

        txt_to_csv(src, dst)

        lines = dst.read_text().splitlines()
        assert len(lines) == 2  # header + 1 valid row


class TestTxtToCsvAligned:
    def test_first_segment_starts_at_zero(self, tmp_path):
        src = tmp_path / "ann.txt"
        dst = tmp_path / "ann.csv"
        src.write_text("0.5\t2.5\tAm\n2.5\t5.0\tC")

        txt_to_csv_aligned(src, dst)

        lines = dst.read_text().splitlines()
        first_row = lines[1]
        assert first_row.startswith("0.000000;")

    def test_subsequent_start_matches_previous_end(self, tmp_path):
        src = tmp_path / "ann.txt"
        dst = tmp_path / "ann.csv"
        src.write_text("0.0\t2.5\tAm\n2.3\t5.0\tC")

        txt_to_csv_aligned(src, dst)

        lines = dst.read_text().splitlines()
        second_start = lines[2].split(";")[0]
        assert second_start == "2.5"
