777539958
*****
Comments:
Minimax, Alphabeta and Expectimax are implemented recursively as per the pseudocode.
the evaluation function is implemented with an attempt to get the successor to abide by a certain strategy, keeping the
largest tiles in the top row and valuing the highest value merges available.
The better evaluation function is calculated using a variety of different qualities.
the evaluation calculates a variety of properties and adds them together in a weighted sum.
merge score - the value of all the merges on the board. This gives weight to boards with more merges, the higher
value the possible merges the better.

empty squares - the number of empty squares, this is important so the algorithm doesnt favor boards that are too
full and thereby get stuck.

monotonicity - how monotonic the rows are from left to right or right to left or how monotonic the columns are
going up and down or down and up. This means the evaluation function favors boards that are monotonic,
which is a good strategy to play by.

Smoothness - smoothness is a measure of the difference between neighboring tiles. The less smooth, the greater
the differences between neighboring tiles the more points are deducted from the board. This is an evaulation of a board
hat goes hand in hadn with monotonicity, it forces a monotonic board to merge so that it can be as smooth as possible.
center_score - placing high valued tiles in the center of the board seems to generally be a cause of problems,
it seems best to keep high valued tiles near to the edges.

placement_score - an excellent strategy for winnning the game is to keep the highest valued tile in a corner,
this allows the monotonicity to be fairly consistent and forces a certain organization on the tiles and the
possible merges. So if the largest value is in one of the corners there is a bonus and if not points are deducted.

All this is added together with weights chosen by instinct and alot of tinkering.

