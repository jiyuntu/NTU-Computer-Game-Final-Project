#ifndef MYAI_INCLUDED
#define MYAI_INCLUDED

#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <time.h>

#include <algorithm>
#include <bitset>
#include <vector>

#define RED 0
#define BLACK 1
#define CHESS_COVER 14
#define CHESS_EMPTY 15
#define COMMAND_NUM 19
#define TIME_LIMIT 9.5

struct ChessBoard {
  std::bitset<32> chessBB[14];
  std::bitset<32> colorBB[2];
  std::bitset<32> emptyBB;
  std::bitset<32> coverBB;
  int board[32];
  int cover_chess[14];
  int red_chess_num, black_chess_num;
  int no_eat_flip;
  int history[512];
  int history_count;
  int hash;
};

struct Entry {
  std::bitset<32> chessBB[14];
  std::bitset<32> coverBB;
  double score;
  int height;
  int exact;  // exact = -1: invalid; 0: lowerbound; 1: exact; 2: upperbound;
  int move;

  Entry() { exact = -1; }
  Entry(std::bitset<32>* _chessBB, std::bitset<32> _coverBB, double _score,
        int _height, int _exact, int _move)
      : coverBB(_coverBB),
        score(_score),
        height(_height),
        exact(_exact),
        move(_move) {
    for (int i = 0; i < 14; i++) {
      chessBB[i] = _chessBB[i];
    }
  }
};

extern int tb_size;
extern Entry transposition_table[2][1 << 22];

class MyAI {
  const char* commands_name[COMMAND_NUM] = {
      "protocol_version",  "name",          "version",
      "known_command",     "list_commands", "quit",
      "boardsize",         "reset_board",   "num_repetition",
      "num_moves_to_draw", "move",          "flip",
      "genmove",           "game_over",     "ready",
      "time_settings",     "time_left",     "showboard",
      "init_board"};

 public:
  MyAI(void);
  ~MyAI(void);

  // commands
  bool protocol_version(const char* data[], char* response);   // 0
  bool name(const char* data[], char* response);               // 1
  bool version(const char* data[], char* response);            // 2
  bool known_command(const char* data[], char* response);      // 3
  bool list_commands(const char* data[], char* response);      // 4
  bool quit(const char* data[], char* response);               // 5
  bool boardsize(const char* data[], char* response);          // 6
  bool reset_board(const char* data[], char* response);        // 7
  bool num_repetition(const char* data[], char* response);     // 8
  bool num_moves_to_draw(const char* data[], char* response);  // 9
  bool move(const char* data[], char* response);               // 10
  bool flip(const char* data[], char* response);               // 11
  bool genmove(const char* data[], char* response);            // 12
  bool game_over(const char* data[], char* response);          // 13
  bool ready(const char* data[], char* response);              // 14
  bool time_settings(const char* data[], char* response);      // 15
  bool time_left(const char* data[], char* response);          // 16
  bool showboard(const char* data[], char* response);          // 17
  bool init_board(const char* data[], char* response);         // 18

 private:
  int agent_color;
  int red_time, black_time;
  ChessBoard main_chessboard;
  const int index32[32] = {31, 0,  1,  5,  2,  16, 27, 6,  3,  14, 17,
                           19, 28, 11, 7,  21, 30, 4,  15, 26, 13, 18,
                           10, 20, 29, 25, 12, 9,  24, 8,  23, 22};
  const char skind[17] = "PCNRMGKpcnrmgkX-";
  double epsilon = 1e-7;
  int HT[2048];

  // masks
  std::bitset<32> pmoves[32], row_mask[8], column_mask[4];
  int random_table[16][32];

  // statistics
  int collision;
  int hit;
  int best;
  int search_cnt;

#ifdef WINDOWS
  clock_t begin;
#else
  struct timeval begin;
#endif

  void initMask();
  void initRandomTable();
  void initBoardState();
  void initBoardState(const char* data[]);
  int convertChessNo(char c);
  void makeMove(ChessBoard* chessboard, const char move[6]);
  void makeMove(ChessBoard* chessboard, const int move, const int chess_no);
  void generateMove(char move[6]);
  double alphaBeta(const ChessBoard chessboard, int* move, const int color,
                   const int depth, double alpha, double beta);
  double negaScout(ChessBoard chessboard, int* move, const int color,
                   const int depth, const int remain_depth, double alpha,
                   double beta, std::vector<int>* pv);
  int expand(ChessBoard chessboard, int* moves, int color);
  double evaluate(ChessBoard* chessboard, int move_count, int color);
  int popLSB(std::bitset<32>* BB);
  int popMSB(std::bitset<32>* BB);
  int expandGun(ChessBoard chessboard, int* moves, int color);
  int getIndex(std::bitset<32>* LSB);
  void print_chessboard(ChessBoard* chessboard, int color);
  bool isDraw(const ChessBoard* chessboard);
  bool isTimeUp();
  bool Referee(const int* chess, const int from_location_no,
               const int to_location_no, const int UserId);
  int zobrist(ChessBoard* chessboard);
  int expandFlip(ChessBoard chessboard, int* flip_moves);
  double star1(ChessBoard chessboard, int pos, int color, int depth,
               int remain_depth, double alpha, double beta, std::vector<int>* pv);
  int definitely_win(ChessBoard* chessboard);
};

#endif
