"""Synthetic, explicitly labelled routing traces with template-disjoint splits."""

import json
import random
from pathlib import Path

FAMILIES = {
    "search": [
        "Find footage showing {subject}", "Show me scenes with {subject}",
        "Search my videos for {subject}", "Which clips contain {subject}?",
        "I need shots of {subject}", "Look for {subject} in my footage",
        "Get the timestamps for {subject}", "Pull up a scene featuring {subject}",
        "Find every appearance of {subject}", "Locate video of {subject}",
        "Show visual matches for {subject}", "Search the frames for {subject}",
        "Track down shots featuring {subject}", "Where in the footage is {subject}?",
        "Can you locate excerpts containing {subject}?", "Return moments depicting {subject}",
    ],
    "library": [
        "List my videos", "Show all uploaded files", "What footage is in my library?",
        "Display the video library", "Which videos have I uploaded?", "Show my media files",
        "Give me a list of the clips", "Open my footage library", "What videos are available?",
        "List the source filenames", "Show me my uploaded videos", "Display all sources",
        "Enumerate my video assets", "What source footage do we have?",
        "Bring up an inventory of the uploads", "Let me browse the available recordings",
    ],
    "count": [
        "How many videos have I uploaded?", "What is the total duration of my footage?",
        "Count the videos in my library", "How long is all my footage combined?",
        "Give me the number of uploaded files", "Calculate the total video duration",
        "How many clips are in this workspace?", "Sum the running time of the videos",
        "Tell me the library statistics", "Count my source clips", "Total the footage length",
        "What is the combined running time?", "Calculate how many source videos we have",
        "Report the size of the video collection", "Add up the runtime across my uploads",
        "Give me a numerical summary of my video library",
    ],
    "export": [
        "Export the selected clip", "Download this scene", "Prepare a clip of this moment",
        "Save the selected video segment", "Make an MP4 of this shot", "Trim this selection",
        "Cut out the highlighted scene", "I want to download this clip", "Export this moment",
        "Create a downloadable excerpt", "Prepare the current selection for download",
        "Give me an MP4 of the selected footage", "Render the selected segment to a file",
        "Can I get this excerpt as a video?", "Package this part of the video for download",
        "Produce a short cut from the selection",
    ],
    "inspect": [
        "Detect objects in this frame", "Identify what is in the selected frame",
        "Draw bounding boxes on this image", "Run object detection here",
        "What objects are visible in this still?", "Analyze the selected frame's objects",
        "Find the objects in this image", "Label the objects on this frame",
        "Inspect the current still image", "Show detected objects in this shot",
        "Recognize objects in the selected image", "Highlight objects in this video frame",
        "Mark the visible objects in this still", "What can the detector identify here?",
        "Apply object recognition to the current frame", "Examine this image with the detector",
    ],
}
SUBJECTS = [
    "a rabbit", "trees", "a red car", "someone walking", "a beach", "a city street",
    "a person cooking", "a dog", "mountains", "a bicycle", "a desk", "water",
    "a bird flying", "flowers", "a person running", "the sky",
]


def build():
    output = {s: [] for s in ["train", "validation", "test"]}
    for plan, families in FAMILIES.items():
        for family, template in enumerate(families):
            split = "train" if family < 12 else "validation" if family < 14 else "test"
            for variant in range(8 if split == "train" else 4):
                subject = SUBJECTS[(family * 3 + variant) % len(SUBJECTS)]
                query = template.format(subject=subject)
                if variant % 4 == 1:
                    query = "Please " + query[0].lower() + query[1:]
                elif variant % 4 == 2:
                    query += " please"
                elif variant % 4 == 3:
                    query += " in this workspace"
                output[split].append({
                    "id": f"{plan}-{family}-{variant}", "template_group": f"{plan}-{family}",
                    "query": query, "selected": variant % 2 == 0, "previous": "",
                    "plan": plan, "source": "synthetic-routing-template",
                    "tools": {
                        "search": ["search_frames", "retrieve_frame"],
                        "library": ["query_library"], "count": ["query_library"],
                        "export": ["request_export"], "inspect": ["detect_objects"],
                    }[plan],
                })
    for split, rows in output.items():
        random.Random(20260106).shuffle(rows)
        path = Path(__file__).parent / f"{split}.jsonl"
        path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return output


if __name__ == "__main__":
    print({split: len(rows) for split, rows in build().items()})
