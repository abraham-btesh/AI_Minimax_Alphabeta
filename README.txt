777539958
*****
Comments:
Minimax, Alphabeta and Expectimax are implemented recursively as per the pseudocode.
the evaluation function is implemented with an attempt to get the successor to abide by a certain strategy, keeping the
largest tiles in the top row and valuing the highest value merges available.
The better evaluation function is calculated using a variety of different qualities. Monotonicity of rows and columns,
smoothness (the difference between neighboring tiles), how many tiles are empty, the value of the available merges. As
well we add points for keeping the largest tile in one of the corners, we deduct points for having particularly large
tiles in the center. More details are to be found in the function description.

