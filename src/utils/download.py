import urllib.request

def show_progress(count, block_size, total_size):
    if count == 0:
        print("Downloading", end='')
    else:
       print("\b\b\b\b\b\b\b\b\b", end='')
    percent = (count*block_size)/total_size
    percent = 100*percent
    if percent<100:
        text = f"{percent:.2f}%"
        while len(text)<9:
            text = " " + text
        print(text, end='')

def download(url, target_path, verbose=True):
    if verbose:
        filename, headers = urllib.request.urlretrieve(url, target_path, show_progress)
    else:
        filename, headers = urllib.request.urlretrieve(url, target_path)

