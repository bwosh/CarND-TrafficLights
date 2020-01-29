import urllib.request

def show_progress(count, block_size, total_size):
    percent = (count*block_size)/total_size
    percent = 100*percent

    text = f"{percent:.2f}%"
    print(f"[{text}]", end='')

def download(url, target_path, verbose=True):
    if verbose:
        filename, headers = urllib.request.urlretrieve(url, target_path, show_progress)
    else:
        filename, headers = urllib.request.urlretrieve(url, target_path)

