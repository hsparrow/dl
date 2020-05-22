import numpy as np

'''
SymbolSets: A list containing all the symbols (the vocabulary without blank)

y_probs: Numpy array with shape (# of symbols + 1, Seq_length, batch_size)
         Your batch size for part 1 will remain 1, but if you plan to use your
         implementation for part 2 you need to incorporate batch_size.

Return the forward probability of the greedy path (a float) and
the corresponding compressed symbol sequence i.e. without blanks
or repeated symbols (a string).
'''


def GreedySearch(SymbolSets, y_probs):
    # Follow the pseudocode from lecture to complete greedy search :-)

    seq_len = y_probs.shape[1]
    idxs, forward_prob, forward_path = [0] * (seq_len + 1), 1., ""
    for t in range(seq_len):
        idx = np.argmax(y_probs[:, t, :])
        if idx == 0:
            forward_path += "-"
        else:
            if not forward_path or SymbolSets[idx - 1] != forward_path[-1]:
                forward_path += SymbolSets[idx - 1]
        forward_prob *= np.max(y_probs[:, t, :])

    return forward_path.replace("-", ""), forward_prob

##############################################################################

'''
SymbolSets: A list containing all the symbols (the vocabulary without blank)

y_probs: Numpy array with shape (# of symbols + 1, Seq_length, batch_size)
         Your batch size for part 1 will remain 1, but if you plan to use your
         implementation for part 2 you need to incorporate batch_size.

BeamWidth: Width of the beam.

The function should return the symbol sequence with the best path score
(forward probability) and a dictionary of all the final merged paths with
their scores.
'''


def InitializePaths(SymbolSet, y):
    InitialPathsWithFinalBlank, InitialPathsWithFinalSymbol = [], []
    InitialBlankPathScore, InitialPathScore = {}, {}

    path = ""
    InitialPathsWithFinalBlank.append(path)
    InitialBlankPathScore[path] = y[0]

    for i, symbol in enumerate(SymbolSet):
        path = symbol
        InitialPathScore[path] = y[i + 1]
        InitialPathsWithFinalSymbol.append(path)

    return InitialPathsWithFinalBlank, InitialPathsWithFinalSymbol, InitialBlankPathScore, InitialPathScore


def Prune(PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore, BeamWidth):
    PrunedBlankPathScore, PrunedPathScore, scorelist = {}, {}, []
    for path in PathsWithTerminalBlank:
        scorelist.append(BlankPathScore[path])
    for path in PathsWithTerminalSymbol:
        scorelist.append(PathScore[path])

    scorelist.sort(reverse=True)
    cutoff = scorelist[BeamWidth - 1] if BeamWidth < len(scorelist) else scorelist[-1]

    PrunedPathsWithTerminalBlank, PrunedPathsWithTerminalSymbol = set(), set()
    for path in PathsWithTerminalBlank:
        if BlankPathScore[path] >= cutoff:
            PrunedPathsWithTerminalBlank.add(path)
            PrunedBlankPathScore[path] = BlankPathScore[path]
    for path in PathsWithTerminalSymbol:
        if PathScore[path] >= cutoff:
            PrunedPathsWithTerminalSymbol.add(path)
            PrunedPathScore[path] = PathScore[path]

    return PrunedPathsWithTerminalBlank, PrunedPathsWithTerminalSymbol, PrunedBlankPathScore, PrunedPathScore


def ExtendWithBlank(PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathSore, PathScore, y):
    UpdatedPathsWithTerminalBlank, UpdatedBlankPathScore = [], {}
    for path in PathsWithTerminalBlank:
        UpdatedPathsWithTerminalBlank.append(path)
        UpdatedBlankPathScore[path] = BlankPathSore[path] * y[0]

    for path in PathsWithTerminalSymbol:
        if path in UpdatedPathsWithTerminalBlank:
            UpdatedBlankPathScore[path] += PathScore[path] * y[0]
        else:
            UpdatedPathsWithTerminalBlank.append(path)
            UpdatedBlankPathScore[path] = PathScore[path] * y[0]
    return UpdatedPathsWithTerminalBlank, UpdatedBlankPathScore


def ExtendWithSymbol(PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore, SymbolSet, y):
    UpdatedPathsWithTerminalSymbol, UpdatedPathScore = [], {}
    for path in PathsWithTerminalBlank:
        for i, c in enumerate(SymbolSet):
            newpath = path + c
            UpdatedPathsWithTerminalSymbol.append(newpath)
            UpdatedPathScore[newpath] = BlankPathScore[path] * y[i + 1]

    for path in PathsWithTerminalSymbol:
        for i, c in enumerate(SymbolSet):
            newpath = path + c if c != path[-1] else path
            if newpath in UpdatedPathsWithTerminalSymbol:
                UpdatedPathScore[newpath] += PathScore[path] * y[i + 1]
            else:
                UpdatedPathsWithTerminalSymbol.append(newpath)
                UpdatedPathScore[newpath] = PathScore[path] * y[i + 1]

    return UpdatedPathsWithTerminalSymbol, UpdatedPathScore


def MergeIdenticalPaths(PathsWithTerminalBlank, BlankPathScore, PathsWithTerminalSymbol, PathScore):
    MergedPaths, FinalPathScore = PathsWithTerminalSymbol, PathScore
    for path in PathsWithTerminalBlank:
        if path in MergedPaths:
            FinalPathScore[path] += BlankPathScore[path]
        else:
            MergedPaths.append(path)
            FinalPathScore[path] = BlankPathScore[path]
    return MergedPaths, FinalPathScore


def BeamSearch(SymbolSets, y_probs, BeamWidth):
    # Follow the pseudocode from lecture to complete beam search :-)
    seq_len = y_probs.shape[1]

    PathScore, BlankPathScore = {}, {}
    NewPathsWithTerminalBlank, NewPathsWithTerminalSymbol, NewBlankPathScore, NewPathScore = InitializePaths(SymbolSets,
                                                                                                             y_probs[:,
                                                                                                             0, :])

    for t in range(1, seq_len):
        PathsWithTerminalBlank, PathsWithTerminalSymbol, BlankPathScore, PathScore = Prune(NewPathsWithTerminalBlank,
                                                                                           NewPathsWithTerminalSymbol,
                                                                                           NewBlankPathScore,
                                                                                           NewPathScore,
                                                                                           BeamWidth)

        NewPathsWithTerminalBlank, NewBlankPathScore = ExtendWithBlank(PathsWithTerminalBlank,
                                                                       PathsWithTerminalSymbol,
                                                                       BlankPathScore,
                                                                       PathScore,
                                                                       y_probs[:, t, :])

        NewPathsWithTerminalSymbol, NewPathScore = ExtendWithSymbol(PathsWithTerminalBlank,
                                                                    PathsWithTerminalSymbol,
                                                                    BlankPathScore,
                                                                    PathScore,
                                                                    SymbolSets,
                                                                    y_probs[:, t, :])

    MergedPaths, FinalPathScore = MergeIdenticalPaths(NewPathsWithTerminalBlank,
                                                      NewBlankPathScore,
                                                      NewPathsWithTerminalSymbol,
                                                      NewPathScore)

    SortedFinalScore = sorted(FinalPathScore.items(), key=lambda x: x[1], reverse=True)

    return SortedFinalScore[0][0], FinalPathScore
