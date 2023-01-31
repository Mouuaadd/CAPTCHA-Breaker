import json


def load_json_data(json_file):
    with open(json_file) as f:
        data = json.load(f)

    # create the datasets
    target_puzzle_piece = []
    target_hole = []

    image_path = data[0]["image"]
    target_puzzle_piece.append(
        (
            data[0]["annotations"][0]["coordinates"]["x"],
            data[0]["annotations"][0]["coordinates"]["y"],
        )
    )
    target_hole.append(
        (
            data[0]["annotations"][1]["coordinates"]["x"],
            data[0]["annotations"][1]["coordinates"]["y"],
        )
    )

    return image_path, target_puzzle_piece, target_hole
