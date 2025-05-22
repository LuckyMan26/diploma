import tensorflow as tf

def parse_record(raw_record):
    """Parses a single TFRecord example and extracts relevant features."""
    example = tf.train.Example()
    example.ParseFromString(raw_record.numpy())
    feat_map = example.features.feature

    filename = feat_map['filename'].bytes_list.value[0].decode()
    alignment_score = feat_map['misalignment_score'].float_list.value[0]
    aesthetics_score = feat_map['aesthetics_score'].float_list.value[0]
    artifact_score = feat_map['artifact_score'].float_list.value[0]

    uid = filename.split(sep="/")[1].split(sep=".")[0]
    
    human_score = {
        #"artifact_score": artifact_score,
        #"aesthetics_score": aesthetics_score,
        "alignment_score": alignment_score
    }

    return filename, uid, human_score


def search_for_image(ds: list[dict], uid : str):
   #ds_words = ds.select_columns(["caption", "best_image_uid", "jpg_0","jpg_1", "image_0_uid", "image_1_uid"])

    for example in ds:
        #print(f"example: {example}")
        if example["best_image_uid"] in uid or \
        example["image_0_uid"] in uid or  \
        example["image_1_uid"] in uid:
            if example["image_0_uid"]==uid:
                #print("0")
                image = example["jpg_0"]
                model = example["model_0"]
            else:
                #print('1')
                image = example["jpg_1"]
                model = example["model_1"]
                
            prompt = example["caption"]

            return image,prompt
    return None,None