import requests
from tqdm import tqdm

import json
import re

from bs4 import BeautifulSoup
from markdown import markdown


def markdown_to_text(markdown_string):
    """ Converts a markdown string to plaintext """

    # md -> html -> text since BeautifulSoup can extract text cleanly
    html = markdown(markdown_string)

    # remove code snippets
    html = re.sub(r'<pre>(.*?)</pre>', ' ', html)
    html = re.sub(r'<code>(.*?)</code >', ' ', html)

    # extract text
    soup = BeautifulSoup(html, "html.parser")
    text = ' '.join(soup.findAll(string=True))

    return text


def replace_urls(x, url_replacement_token='<URL>'):
    return re.sub("http(.+)?(\W|$)", url_replacement_token, x)


def replace_ss_prefix(x):
    return re.sub(r'^\W*(summary statement|submission statement|ss)[^a-zA-Z]*',"",  x, flags=re.I|re.U).strip()

def preprocess(x):
    return replace_ss_prefix(replace_urls(markdown_to_text(x)))


def rehydrate_comments(input_file, output_file):
    """
    Rehydrates comment bodies from Reddit using the arctic-shift endpoint
    and saves the data in a JSONL format similar to the input.

    Args:
        input_file (str): Path to the JSONL file containing comment _ids (with 't1_' prefix).
        output_file (str): Path to the JSONL file to be created with rehydrated data.
    """
    ids_to_fetch_with_prefix = []
    original_data_map = {}

    with open(input_file, 'r') as f:
        for line in f:
            try:
                item = json.loads(line)
                if '_id' in item and item['_id'].startswith('t1_'):
                    comment_id_with_prefix = item['_id']
                    comment_id_without_prefix = comment_id_with_prefix[3:] # Remove 't1_' prefix
                    ids_to_fetch_with_prefix.append(comment_id_with_prefix)
                    original_data_map[comment_id_with_prefix] = item # Store original data by full _id
            except json.JSONDecodeError:
                print(f"Skipping invalid JSON line: {line.strip()}")

    base_url = "https://arctic-shift.photon-reddit.com/api/comments/ids"
    fields = "body,subreddit,id"
    rehydrated_data = []

    # Process IDs in batches of up to 500
    for i in tqdm(range(0, len(ids_to_fetch_with_prefix), 500), desc="Rehydrating comments"):
        batch_ids_with_prefix = ids_to_fetch_with_prefix[i:i + 500]
        batch_ids_without_prefix = [comment_id[3:] for comment_id in batch_ids_with_prefix] # Remove prefix for the request
        params = {
            "ids": ",".join(batch_ids_without_prefix),
            "fields": fields
        }

        try:
            response = requests.get(base_url, params=params)
            response.raise_for_status()  # Raise an exception for bad status codes
            api_response = response.json()

            if "data" in api_response and isinstance(api_response["data"], list):
                rehydrated_comments_batch = api_response["data"]
                rehydrated_map = {comment['id']: comment for comment in rehydrated_comments_batch if comment.get('body', '[deleted]').strip() not in ['[deleted]', '[removed]']}

                for comment_id_with_prefix in batch_ids_with_prefix:
                    comment_id_without_prefix = comment_id_with_prefix[3:]
                    if comment_id_without_prefix in rehydrated_map and comment_id_with_prefix in original_data_map:
                        rehydrated_comment = rehydrated_map[comment_id_without_prefix]
                        original_item = original_data_map[comment_id_with_prefix]
                        merged_item = {
                            "_id": f"t1_{rehydrated_comment['id']}", # Add the prefix back to the _id in the output
                            "text": preprocess(rehydrated_comment['body']),
                            "subreddit": rehydrated_comment['subreddit'],
                            "conspiracy": original_item.get("conspiracy"),
                            "markers": original_item.get("markers"),
                            "annotator": original_item.get("annotator")
                        }
                        rehydrated_data.append(merged_item)
                    elif comment_id_with_prefix in original_data_map:
                        print(f"Warning: Could not rehydrate comment with id '{comment_id_with_prefix}'")
            else:
                print(f"Warning: Unexpected API response format for batch starting with '{batch_ids_without_prefix[0]}'")

        except requests.exceptions.RequestException as e:
            print(f"Error during API request: {e}")
            print(f"Failed to rehydrate batch starting with id '{batch_ids_without_prefix[0]}'")
            continue
        except json.JSONDecodeError:
            print(f"Error decoding JSON response for batch starting with id '{batch_ids_without_prefix[0]}'")
            continue

    # Save the rehydrated data to the output file
    with open(output_file, 'w') as outfile:
        for item in rehydrated_data:
            outfile.write(json.dumps(item) + '\n')

    print(f"Rehydrated data saved to '{output_file}'")


if __name__ == "__main__":

    for input_file, output_file in [("train_redacted.jsonl", "train_rehydrated.jsonl"),
                                    ("dev_redacted.jsonl", "dev_rehydrated.jsonl")]:
        rehydrate_comments(input_file, output_file)
