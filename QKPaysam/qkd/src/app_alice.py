from netqasm.sdk.external import NetQASMConnection, Socket
from netqasm.sdk import EPRSocket
from Block import Block, Block_List
import numpy as np
import random

N = 200


def main(app_config=None, x=0, y=0):

    def S(diffBases):

        def finder_p(a, b):
            total = sum(diffBases[:, 0] == a & (diffBases[:, 1] == b))
            p_p = (sum(diffBases[:, 0] == a & (diffBases[:, 1] == b) & (diffBases[:, 2] == 1) & (
                diffBases[:, 2] == 1)) +
                    sum(diffBases[:, 0] == a & (diffBases[:, 1] == b) & (diffBases[:, 2] == -1) & (
                            diffBases[:, 3] == -1))) / max(1, total)
            p_n = (sum(diffBases[:, 0] == a & (diffBases[:, 1] == b) & (diffBases[:, 2] == -1) & (
                diffBases[:, 2] == 1)) +
                    sum(diffBases[:, 0] == a & (diffBases[:, 1] == b) & (diffBases[:, 2] == 1) & (
                            diffBases[:, 3] == -1))) / max(1, total)

            return p_p - p_n

        exp_a1b1 = finder_p(1, 1)
        exp_a1b3 = finder_p(1, 3)
        exp_a3b1 = finder_p(3, 1)
        exp_a3b3 = finder_p(3, 3)
        print(exp_a1b1, exp_a1b3, exp_a3b1, exp_a3b3)

        return exp_a1b1 - exp_a1b3 + exp_a3b1 + exp_a3b3

    # Specify an EPR socket to bob
    epr_socket = EPRSocket("bob")
    # Setup a classical socket to bob
    socket = Socket("alice", "bob", log_config=app_config.log_config)

    alice_outputs = []

    alice = NetQASMConnection(
        "alice",
        log_config=app_config.log_config,
        epr_sockets=[epr_socket],
    )
    with alice:
        alice_basis = []
        for _ in range(N):
            base = random.randint(0,2)
            alice_basis.append(str(base))
            # Create an entangled pair using the EPR socket to bob
            q_ent = epr_socket.create()[0]
            # Measure the qubit
            q_ent.rot_Y(n=base, d=2)
            m = q_ent.measure()
            alice.flush()
            alice_outputs.append(str(m))
    print('Alice basis are:', alice_basis)
    #Alice is sending to Bob her basis
    socket.send("".join(alice_basis))
    #Alice receives the basis from Bob
    bob_basis_received = socket.recv()
    bob_basis_received = [int(i) for i in bob_basis_received]
    print(f"alice received the bob basis: {bob_basis_received}")

    alice_basis = np.array(alice_basis).astype(int)
    bob_basis = np.array(bob_basis_received).astype(int)

    mask_sift = np.equal(alice_basis, bob_basis + np.ones(len(bob_basis), dtype=int))

    alice_outputs_chsh = np.array([alice_outputs[ii] for ii in range(len(alice_outputs))])[np.invert(mask_sift)]
    socket.send("".join(alice_outputs_chsh))
    bob_outputs_chsh = socket.recv()

    alice_outputs_chsh = np.array(alice_outputs_chsh).astype(int)
    alice_basis_chsh = np.array(alice_basis)[np.invert(mask_sift)]
    bob_outputs_chsh = np.array([bob_outputs_chsh[ii] for ii in range(len(bob_outputs_chsh))]).astype(int)
    bob_basis_chsh = np.array(bob_basis)[np.invert(mask_sift)]

    # move to {-1,1} encoding
    alice_outputs_chsh = 2 * alice_outputs_chsh - 1
    bob_outputs_chsh = 2 * bob_outputs_chsh - 1

    # print('aoc', alice_outputs_chsh)
    # print('abc', alice_basis_chsh)
    # print('boc', bob_outputs_chsh)
    # print('bbc', bob_basis_chsh)

    total_data_chsh = np.transpose(np.vstack((
        alice_basis_chsh,
        bob_basis_chsh,
        alice_outputs_chsh,
        bob_outputs_chsh
    )))

    Alice_key = np.array(alice_outputs)[mask_sift]

    #part where the CASCADE protocol comes in
    size = int(len(Alice_key))
    p_error = 0.2
    max_passes = 4
    # Alice receives the seed 
    seed = int(socket.recv())
    np.random.seed(seed)
    #compute an array of random positions
    pos_array = np.arange(0, len(Alice_key))
    # Definition of parameters according to original cascade
    k1 = int(np.ceil(0.73/p_error))
    k = [k1, 2*k1, 4*k1, 8*k1]
    list_of_alice_blocks = Block_List()
    
    # Cycle of passes
    for index in range(max_passes):
        alice_parities = []
        if index > 0:
            # shuffle the array of positions. 
            np.random.shuffle(pos_array)
        # Split the array of positions into blocks
        masks = np.split(pos_array, np.arange(k[index], size, k[index]))
        for m in masks:
            # Bob compute blocks
            alice_block = Block(Alice_key, m)
            alice_parities.append(str(int(alice_block.parity)))
            list_of_alice_blocks.append(alice_block)

        bob_parities = [str(int(s)) for s in socket.recv()]
        #transform bob parities to string in order to send them
        alice_parities_str = "".join(alice_parities)
        socket.send(alice_parities_str)

    while list_of_alice_blocks.needs_correction():
        # We find the smallest block in the list (previous and present passes) that has true relative parity
        best_index = list_of_alice_blocks.shortest_problem()
        # Alice must communicate to Bob the index of the Block
        bits_communicated += np.log2(best_index+1)
        # We perform binary search on that block
        position, waste = binary_search(list_of_alice_blocks[best_index], list_of_bob_blocks[best_index])
        # waste is the amount of bits communicated in binary search (approx log2(block.size))
        bits_communicated += waste
        bits_published += waste
        # We correct the problematic bit
        Alice_key[position] = not Alice_key[position]
        # We flip parity and relative parity for all alice's blocks that contain the bit
        for block in list_of_alice_blocks:
            if block.contains(position):
                block.flip_parities()

    return {
        "secret_key": Alice_key,
    }
