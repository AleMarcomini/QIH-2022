from netqasm.sdk.external import NetQASMConnection, Socket
from netqasm.sdk import EPRSocket
import numpy as np
import random

N = 10000

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

    mask_sift = np.equal(alice_basis, bob_basis - np.ones(len(bob_basis)))

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

    # print(S(total_data_chsh))


    # nMeasurments = 100
    #
    # def createGroup(expOutput):
    #     diffBases = expOutput[expOutput[:, 0] != expOutput[:, 1], :]
    #     sameBases = expOutput[expOutput[:, 0] == expOutput[:, 1] & (expOutput[:, 0] != 3) & (expOutput[:, 1] != 3), :]
    #
    #     return (diffBases, sameBases)
    #
    # aBases = np.random.choice([1, 2, 3], nMeasurments, p=[1 / 3, 1 / 3, 1 / 3])
    # bBases = np.random.choice([1, 2, 3], nMeasurments, p=[1 / 3, 1 / 3, 1 / 3])
    # aOutcome = np.random.choice([1, -1], nMeasurments, p=[0.5, 0.5])
    # bOutcome = np.random.choice([1, -1], nMeasurments, p=[0.5, 0.5])
    #
    # expOutput = np.transpose(np.vstack((aBases, bBases, aOutcome, bOutcome)))
    #
    # diffBases, sameBases = createGroup(expOutput)
    #
    # S = S(diffBases)
    #
    # print(S)
    # # baseA,baseB,outcomeA,outcomeB
    #
    #
    # # Compute S
    # diffBases = np.transpose(np.vstack((aBases, bBases, aOutcome, bOutcome)))