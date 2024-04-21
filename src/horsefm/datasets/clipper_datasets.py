""" """

from collections import defaultdict
import dataclasses
import os
from typing import Dict, Set

import pandas

from .. import common
from .. import clipper
from ..datasets import clipper_mlp_values
from ..datasets import clipper_other_values


class ClipperParamsHelper(clipper.ClipperParams):
    _rootdir: str
    known_characters: Dict
    known_tags: Dict
    known_noise_levels: Dict
    known_sources: Dict
    unknown_characters: Set = set()
    unknown_tags: Set = set()
    unknown_noise_levels: Set = set()
    unknown_sources: Set = set()

    def __init__(
        self, masterfile_1, masterfile_2, characters=None, tags=None, noise_levels=None, sources=None
    ):
        self.rootdir1 = masterfile_1
        self.rootdir2 = masterfile_2
        self.known_characters = characters or {}
        self.known_tags = tags or {}
        self.known_noise_levels = noise_levels or {}
        self.known_sources = sources or {}


    def relpath(self, filepath: str):
        normpath = os.path.normpath(filepath)

        if normpath.startswith(self.rootdir1):
            rootdir = 'masterfile1/'
            rootlen = len(self.rootdir1)
        elif normpath.startswith(self.rootdir2):
            rootdir = 'masterfile2/'
            rootlen = len(self.rootdir2)
        else:
            raise ValueError(
                'cannot take the relative path of "{}" since it is not under "{}"'.format(
                    normpath, self.rootdir1
                )
            )

        if normpath == rootdir:
            return ""

        return rootdir + normpath[rootlen + 1:]

    def characters(self, candidate, path):
        if candidate in self.known_characters:
            return self.known_characters[candidate]

        if candidate not in self.unknown_characters:
            print("unknown character", candidate)
            self.unknown_characters.add(candidate)

        # default result
        return candidate

    def tags(self, candidate, path):
        if candidate in self.known_tags:
            return self.known_tags[candidate]

        if candidate not in self.unknown_tags:
            print("unknown tag", candidate)
            self.unknown_tags.add(candidate)

        # default result
        return candidate

    def noise_levels(self, candidate, path):
        if candidate in self.known_noise_levels:
            return self.known_noise_levels[candidate]

        if candidate not in self.unknown_noise_levels:
            print("unknown noise level", candidate)
            self.unknown_noise_levels.add(candidate)

        # default result
        return candidate

    def sources(self, candidate, path):
        candidate = self.relpath(candidate)

        if candidate in self.known_sources:
            return self.known_sources[candidate]

        if candidate not in self.unknown_sources:
            print("unknown source", candidate, "for", path)
            self.unknown_sources.add(candidate)

        # default result
        return candidate


class MlpDialogueParams(ClipperParamsHelper):
    def __init__(self, masterfile_1, masterfile_2):
        super(MlpDialogueParams, self).__init__(
            masterfile_1,
            masterfile_2,
            characters=clipper_mlp_values.CHARACTERS,
            tags=clipper_mlp_values.TAGS,
            noise_levels=clipper_mlp_values.NOISE,
        )

    def sources(self, candidate, path):
        relpath = self.relpath(candidate)

        if relpath.endswith("labels.txt"):
            relpath = os.path.dirname(relpath)

        if relpath in clipper_mlp_values.SOURCES:
            return clipper_mlp_values.SOURCES[relpath]

        if candidate not in self.unknown_sources:
            print("unknown source", candidate, "for", path)
            self.unknown_sources.add(candidate)

        # default result
        return candidate

class MlpSfxParams(ClipperParamsHelper):
    def __init__(self, masterfile_1, masterfile_2):
        super(MlpSfxParams, self).__init__(
            masterfile_1,
            masterfile_2,
            characters=clipper_mlp_values.CHARACTERS,
            tags=clipper_mlp_values.TAGS,
            noise_levels=clipper_mlp_values.NOISE,
        )

    def sources(self, candidate, path):
        relpath = self.relpath(candidate)

        if relpath.endswith("labels.txt"):
            relpath = os.path.dirname(relpath)

        if relpath in clipper_mlp_values.SOURCES:
            return clipper_mlp_values.SOURCES[relpath]

        if candidate not in self.unknown_sources:
            print("unknown source", candidate, "for", path)
            self.unknown_sources.add(candidate)

        # default result
        return candidate


def mlp_dialogue_dataset(masterfile_1, masterfile_2):
    params = MlpDialogueParams(masterfile_1, masterfile_2)
    dataset = clipper.ClipperSet(params)

    print("warning: ignoring ponysorter files")
    #for entry in os.scandir(f"{clipper_root}/Reviewed episodes"):
        #if not entry.is_file():
            #print(f"Unexpected directory: Reviewed episodes/{entry.name}")
            #continue
        # dataset.load_ponysorter(entry.path)
        #pass

    clip_directories = [
        f"{masterfile_1}/Sliced Dialogue/EQG",
        f"{masterfile_1}/Sliced Dialogue/FiM",
        #f"{clipper_root}/Sliced Dialogue/MLP Movie",
        f"{masterfile_1}/Sliced Dialogue/Special source",
        # f"{clipper_root}/Sliced Dialogue/Other/Mobile game",
    ]

    if masterfile_2 != None:
        clip_directories.extend([
            f"{masterfile_2}/Songs",
        ])


    for clip_directory in clip_directories:
        print("loading", clip_directory)
        for root, dirs, files in os.walk(clip_directory):
            if root.endswith('/Source files'):
                continue
            for filename in files:
                if filename == "labels.txt":
                    continue

                if filename.endswith(".txt"):
                    dataset.load_transcript(f"{root}/{filename}")
                elif filename.endswith(".flac"):
                    dataset.load_audio(f"{root}/{filename}")
                else:
                    print(f"Unexpected file: {root}/{filename}")

    for entry in os.scandir(f"{masterfile_1}/Sliced Dialogue/Label files"):
        if entry.name.endswith("_sfx.txt"):
            continue
        if entry.name.endswith('_music.txt'):
            continue
        if not entry.is_file():
            continue
        if not entry.name.endswith(".txt"):
            continue

        dataset.load_audacity(entry.path)

    return dataset



def mlp_sfx_params(masterfile_1, masterfile_2):
    # todo: convert m4a to flac
    # todo: restructure sfx sounds
    # ... source (e.g., Hoers, Rain, character) / Tag number.flac

    # todo: allow manually adding tags/records to a clipperset
    # maybe post_audio vs put_audio

    clip_directories = [
        f"{masterfile_2}/SFX and Music/Music",
        f"{masterfile_2}/SFX and Music/SFX",
    ]

    audio_files = defaultdict(dict)

    for clip_directory in clip_directories:
        print("loading", clip_directory)
        for root, dirs, files in os.walk(clip_directory):
            for filename in files:
                print('parsing', filename)
                if filename.endswith(".flac"):
                    name = os.path.splitext(filename)[0]
                    parts = name.split('~')
                    audio_files[name] = {
                        'name': name,
                        'path': os.path.join(root, filename),
                        'tags': [x.strip() for x in name.split('~')[0].split(',')],
                        'source': name.split('~')[1].split('-')[0].strip() if len(parts) > 1 else None,
                    }
                else:
                    print(f"Unexpected file: {root}/{filename}")

    for entry in os.scandir(f"{masterfile_1}/Sliced Dialogue/Label files"):
        if not entry.is_file():
            continue
        if not entry.name.endswith("_sfx.txt") and not entry.name.endswith('_music.txt'):
            continue

        with open(entry.path, 'r', encoding='iso-8859-1') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                start, end, name = line.split('\t')

                # if name not in audio_files:
                #     print(f"Unknown audio file: {name}")
                #     continue

                audio_files[name].update({
                    'name': name,
                    'start': start,
                    'end': end,
                })

    return pandas.DataFrame(audio_files.values())


class ExtraDialogueParams(ClipperParamsHelper):
    def __init__(self, clipper_root):
        super(ExtraDialogueParams, self).__init__(
            clipper_root,
            characters=clipper_other_values.CHARACTERS,
            tags=clipper_other_values.TAGS,
            noise_levels=clipper_other_values.NOISE,
            sources=clipper_other_values.SOURCES,
        )


def extra_dialogue_dataset(clipper_root):
    params = ExtraDialogueParams(clipper_root)
    dataset = clipper.ClipperSet(params)

    clip_directories = [
        f"{clipper_root}/Sliced Dialogue/Other/A Little Bit Wicked (Kristin Chenoworth, Skystar)",
        f"{clipper_root}/Sliced Dialogue/Other/ATHF",
        f"{clipper_root}/Sliced Dialogue/Other/CGP Grey",
        f"{clipper_root}/Sliced Dialogue/Other/Dan vs",
        f"{clipper_root}/Sliced Dialogue/Other/Dr. Who",
        f"{clipper_root}/Sliced Dialogue/Other/Eli, Elite Dangerous (John de Lancie, Discord)",
        f"{clipper_root}/Sliced Dialogue/Other/Star Trek (John de Lancie, Discord)",
        f"{clipper_root}/Sliced Dialogue/Other/Sum - Tales From the Afterlives (Emily Blunt, Tempest)/Sum - Tales From the Afterlives (44.1 kHz)",
        f"{clipper_root}/Sliced Dialogue/Other/TFH/",
    ]

    for clip_directory in clip_directories:
        for root, _, files in os.walk(clip_directory):
            for filename in files:
                if filename in (
                    "Converted.txt",
                    "Converted (1).txt",
                    "Note on these Discord lines.txt",
                    "Dr. Who Dictionary.txt",
                ):
                    continue

                if filename.endswith(".txt"):
                    dataset.load_transcript(f"{root}/{filename}")
                elif filename.endswith(".flac"):
                    dataset.load_audio(f"{root}/{filename}")
                else:
                    print(f"Unexpected file: {root}/{filename}")

    for root, _, files in os.walk(f"{clipper_root}/Sliced Dialogue/Label files/Other"):
        for file in files:
            if not file.endswith(".txt"):
                continue

            filepath = os.path.join(root, file)
            dataset.load_audacity(filepath)

    return dataset
