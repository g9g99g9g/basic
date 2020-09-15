###########################################################################
# Co-Author: Shudong YANG
# Update: Sep. 15, 2020
# based on <https://github.com/llSourcell/Simple_Blockchain_in_5_Minutes/>
###########################################################################

import datetime  # generates timestamps
import hashlib  # contains hashing algorithms

# defining the 'block' data structure, each block has 7 attributes
class Block:
    blockNo = 0  # 1 number of the block
    data = None  # 2 what data is stored in this block
    next = None  # 3 pointer to the next block
    hash = None  # 4 the hash of this block
    nonce = 0  # 5 A nonce is a number only used once
    previous_hash = 0x0  # 6 store the hash (ID) of the previous block in the chain
    timestamp = datetime.datetime.now()  # 7 timestamp

    # initialize a block by storing some data in it
    def __init__(self, data):
        self.data = data

    # Function to compute 'hash' of a block
    # a hash acts as both a unique identifier & verifies its integrity. if someone changes the hash of a block, then every block that comes after it is changed, this helps make a blockchain immutable
    def hash(self):
        # SHA-256 is a hashing algorithm that generates an almost-unique 256-bit signature that represents some piece of text
        h = hashlib.sha256()
        # the input to the SHA-256 algorithm will be a concatenated string consisting of 5 block attributes
        # the nonce, data, previous hash, timestamp, & block
        h.update(
            str(self.nonce).encode('utf-8') +
            str(self.data).encode('utf-8') +
            str(self.previous_hash).encode('utf-8') +
            str(self.timestamp).encode('utf-8') +
            str(self.blockNo).encode('utf-8')
        )
        return h.hexdigest()  # returns a hexademical string

    def __str__(self):
        # print out the value of a block
        return "Block Hash: " + str(self.hash()) + \
               "\nBlockNo: " + str(self.blockNo) + \
               "\nBlock Data: " + str(self.data) + \
               "\nNonce: " + str(self.nonce) + "\n" + "-"*50

# defining the blockchain datastructure consists of 'blocks' linked together to form a 'chain'.
class Blockchain:
    diff = 20  # set the mining difficulty
    maxNonce = 2 ** 32  # store in a 32-bit number
    target = 2 ** (256 - diff)  # target hash, for mining
    block = Block("Genesis")  # generates the first block(Genesis Block) in the blockchain
    head = block  # sets it as the head of our blockchain

    # adds a given block to the chain of blocks. the block to be added is the only parameter
    def add(self, block):
        # set the hash of a given block as our new block's previous hash
        block.previous_hash = self.block.hash()
        # set the block of our new block as the given block's # + 1, since its next in the chain
        block.blockNo = self.block.blockNo + 1

        # set the next block equal to itself. This is the new head of the blockchain
        self.block.next = block
        self.block = self.block.next

    # Determines whether or not we can add a given block to the blockchain
    def mine(self, block):
        # from 0 to 2^32
        for n in range(self.maxNonce):
            # is the value of the given block's hash less than our target value?
            if int(block.hash(), 16) <= self.target:
                # if it is, add the block to the chain
                self.add(block)
                print(block)
                break
            else:
                block.nonce += 1

blockchain = Blockchain()  # initialize blockchain

for n in range(10):
    blockchain.mine(Block("Block " + str(n + 1)))  # mine 10 blocks

while blockchain.head != None:
    print(blockchain.head)  # print out each block in the blockchain
    blockchain.head = blockchain.head.next
