import heapq


class Node:
    def __init__(self, char: str | None, freq: int):
        self.char: str | None = char
        self.freq: int = freq
        self.left: Node | None = None
        self.right: Node | None = None

    'Type hinting for "classname" added in 3.11'

    def __lt__(self, other: "Node") -> bool:
        return self.freq < other.freq


def build_huffman_tree(frequency: dict[str, int]) -> Node:
    heap = [Node(char, freq) for char, freq in frequency.items()]
    heapq.heapify(heap)

    while len(heap) > 1:
        left = heapq.heappop(heap)
        right = heapq.heappop(heap)
        print(left.freq, right.freq)
        merged = Node(None, left.freq + right.freq)
        merged.left = left
        merged.right = right
        heapq.heappush(heap, merged)

    return heap[0]


def generate_codes(node: Node | None, prefix: str, codebook: dict[str | None, str]):
    if node is not None:
        if node.char is not None:
            codebook[node.char] = prefix
        _ = generate_codes(node.left, prefix + "0", codebook)
        _ = generate_codes(node.right, prefix + "1", codebook)
    return codebook


chars = ["a", "b", "c", "d", "e"]
frequency = {"a": 24, "b": 12, "c": 10, "d": 8, "e": 8}

root = build_huffman_tree(frequency)
huffman_codes = generate_codes(root, "", {})
print("Character Huffman Codes:")
for char in chars:
    print(f"{char}: {huffman_codes[char]}")
