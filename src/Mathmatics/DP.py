def DP(W,H,G:list):
    board = [[1000] * (W + 1) for i in range(H + 1)]
    for i in range(0, H + 1):
        for j in range(0, W + 1):
            
    return board

if __name__ == "__main__":
    W = int(input())
    H = int(input())
    G = [0, 0]
    result = DP(W, H, G)
    
    for i in range(H + 1):
        print(result[i])