#include "MyAI.h"

#include "float.h"

#define MAX_DEPTH 3

#define WIN 1.0
#define DRAW 0.2
#define LOSE 0.0
#define BONUS 0.3
#define BONUS_MAX_PIECE 8

#define OFFSET (WIN + BONUS)

#define NOEATFLIP_LIMIT 60
#define POSITION_REPETITION_LIMIT 3

#include <random>
std::mt19937 rng(880301);
int randint(int lb, int ub) {
  return std::uniform_int_distribution<int>(lb, ub)(rng);
}

MyAI::MyAI(void) { srand(time(NULL)); }

MyAI::~MyAI(void) {}

bool MyAI::protocol_version(const char* data[], char* response) {
  strcpy(response, "1.1.0");
  return 0;
}

bool MyAI::name(const char* data[], char* response) {
  strcpy(response, "Wendy");
  return 0;
}

bool MyAI::version(const char* data[], char* response) {
  strcpy(response, "1.0.0");
  return 0;
}

bool MyAI::known_command(const char* data[], char* response) {
  for (int i = 0; i < COMMAND_NUM; i++) {
    if (!strcmp(data[0], commands_name[i])) {
      strcpy(response, "true");
      return 0;
    }
  }
  strcpy(response, "false");
  return 0;
}

bool MyAI::list_commands(const char* data[], char* response) {
  for (int i = 0; i < COMMAND_NUM; i++) {
    strcat(response, commands_name[i]);
    if (i < COMMAND_NUM - 1) {
      strcat(response, "\n");
    }
  }
  return 0;
}

bool MyAI::quit(const char* data[], char* response) {
  fprintf(stderr, "Bye\n");
  return 0;
}

bool MyAI::boardsize(const char* data[], char* response) {
  fprintf(stderr, "BoardSize: %s x %s\n", data[0], data[1]);
  return 0;
}

bool MyAI::reset_board(const char* data[], char* response) {
  this->red_time = -1;    // unknown
  this->black_time = -1;  // unknown
  this->initBoardState();
  this->game_elapsed = 0;
  this->first_play = 1;
  return 0;
}

bool MyAI::num_repetition(const char* data[], char* response) { return 0; }

bool MyAI::num_moves_to_draw(const char* data[], char* response) { return 0; }

bool MyAI::move(const char* data[], char* response) {
  char move[6];
  sprintf(move, "%s-%s", data[0], data[1]);
  this->makeMove(&(this->main_chessboard), move);
  return 0;
}

bool MyAI::flip(const char* data[], char* response) {
  char move[6];
  sprintf(move, "%s(%s)", data[0], data[1]);
  this->makeMove(&(this->main_chessboard), move);
  return 0;
}

bool MyAI::genmove(const char* data[], char* response) {
  // set color
  if (!strcmp(data[0], "red")) {
    this->agent_color = RED;
  } else if (!strcmp(data[0], "black")) {
    this->agent_color = BLACK;
  } else {
    this->agent_color = 2;
  }
  // genmove
  char move[6];
  this->generateMove(move);
  sprintf(response, "%c%c %c%c", move[0], move[1], move[3], move[4]);
  return 0;
}

bool MyAI::game_over(const char* data[], char* response) {
  fprintf(stderr, "Game Result: %s\n", data[0]);
  return 0;
}

bool MyAI::ready(const char* data[], char* response) { return 0; }

bool MyAI::time_settings(const char* data[], char* response) { return 0; }

bool MyAI::time_left(const char* data[], char* response) {
  if (!strcmp(data[0], "red")) {
    sscanf(data[1], "%d", &(this->red_time));
  } else {
    sscanf(data[1], "%d", &(this->black_time));
  }
  fprintf(stderr, "Time Left(%s): %s\n", data[0], data[1]);
  return 0;
}

bool MyAI::showboard(const char* data[], char* response) {
  print_chessboard(&(this->main_chessboard), this->agent_color);
  return 0;
}

bool MyAI::init_board(const char* data[], char* response) {
  initBoardState(data);
  return 0;
}

// *********************** AI FUNCTION *********************** //

int MyAI::zobrist(ChessBoard* chessboard) {
  int ret = 0;
  for (int i = 0; i < 32; i++) {
    ret ^= random_table[chessboard->board[i]][i];
  }
  return ret;
}
void MyAI::initMask() {
  row_mask[0] = std::bitset<32>(0xf);
  row_mask[1] = std::bitset<32>(0xf0);
  row_mask[2] = std::bitset<32>(0xf00);
  row_mask[3] = std::bitset<32>(0xf000);
  row_mask[4] = std::bitset<32>(0xf0000);
  row_mask[5] = std::bitset<32>(0xf00000);
  row_mask[6] = std::bitset<32>(0xf000000);
  row_mask[7] = std::bitset<32>(0xf0000000);

  column_mask[0] = std::bitset<32>(0x11111111);
  column_mask[1] = std::bitset<32>(0x22222222);
  column_mask[2] = std::bitset<32>(0x44444444);
  column_mask[3] = std::bitset<32>(0x88888888);

  for (int i = 0; i < 32; i++) {
    pmoves[i] = std::bitset<32>(0);
    int moves[4] = {i - 1, i + 1, i - 4, i + 4};
    for (int k = 0; k < 4; k++) {
      if (moves[k] >= 0 && moves[k] < 32 &&
          abs(i / 4 - moves[k] / 4) + abs(i % 4 - moves[k] % 4) == 1) {
        pmoves[i][moves[k]] = 1;
      }
    }
  }
}
void MyAI::initRandomTable() {
  for (int i = 0; i < 15; i++) {
    for (int j = 0; j < 32; j++) {
      random_table[i][j] = randint(1, (1 << tb_size) - 1);
    }
  }
  for (int j = 0; j < 32; j++) {
    random_table[CHESS_EMPTY][j] = 0;
  }
}
void MyAI::initBoardState() {
  initMask();
  initRandomTable();

  for (int i = 0; i < 14; i++) {
    main_chessboard.chessBB[i] = std::bitset<32>(0);
  }
  main_chessboard.colorBB[0] = main_chessboard.colorBB[1] = std::bitset<32>(0);
  main_chessboard.emptyBB = std::bitset<32>(0);
  main_chessboard.coverBB = std::bitset<32>(0xffffffff);
  for (int i = 0; i < 32; i++) {
    main_chessboard.board[i] = CHESS_COVER;
  }
  int iPieceCount[14] = {5, 2, 2, 2, 2, 2, 1, 5, 2, 2, 2, 2, 2, 1};
  memcpy(main_chessboard.cover_chess, iPieceCount, sizeof(int) * 14);
  main_chessboard.red_chess_num = 16;
  main_chessboard.black_chess_num = 16;
  main_chessboard.no_eat_flip = 0;
  main_chessboard.history_count = 0;
  main_chessboard.hash = zobrist(&main_chessboard);

  for (int i = 0; i < (1 << tb_size); i++) {
    transposition_table[0][i].exact = -1;
    transposition_table[1][i].exact = -1;
  }
}
void MyAI::initBoardState(const char* data[]) { initMask(); }
int MyAI::convertChessNo(char c) {
  for (int i = 0; i < 14; i++) {
    if (skind[i] == c) return i;
  }
  return -1;
}
void MyAI::makeMove(ChessBoard* chessboard, const char move[6]) {
  int src, dst, m;
  src = ('8' - move[1]) * 4 + (move[0] - 'a');
  if (move[2] == '(') {  // flip
    m = (src << 5) + src;
    printf("# call flip(): flip(%d,%d) = %c\n", src, src, move[3]);
  } else {  // move
    dst = ('8' - move[4]) * 4 + (move[3] - 'a');
    m = (src << 5) + dst;
    printf("# call move(): move : %d-%d \n", src, dst);
  }
  makeMove(chessboard, m, convertChessNo(move[3]));
}
void MyAI::makeMove(ChessBoard* chessboard, const int move,
                    const int flip_chess_no) {
  int src = move >> 5, dst = move & 31;
  int src_no = chessboard->board[src], dst_no = chessboard->board[dst];
  if (src == dst) {
    chessboard->chessBB[flip_chess_no][dst] = 1;
    chessboard->colorBB[flip_chess_no / 7][dst] = 1;
    chessboard->coverBB[dst] = 0;
    chessboard->board[dst] = flip_chess_no;
    chessboard->cover_chess[flip_chess_no]--;
    chessboard->no_eat_flip = 0;
    chessboard->hash ^=
        random_table[CHESS_COVER][dst] ^ random_table[flip_chess_no][dst];
    // if(chessboard->hash != zobrist(chessboard)) fprintf(stderr, "not
    // equal\n");
  } else {
    if (dst_no != CHESS_EMPTY) {  // capturing move
      if (dst_no / 7 == RED)
        chessboard->red_chess_num--;
      else
        chessboard->black_chess_num--;
      chessboard->no_eat_flip = 0;
      chessboard->chessBB[dst_no][dst] = 0;
      chessboard->colorBB[dst_no / 7][dst] = 0;
      chessboard->hash ^= random_table[dst_no][dst];
    } else {
      chessboard->no_eat_flip += 1;
    }
    std::bitset<32> fromToBB(0);
    fromToBB[src] = fromToBB[dst] = 1;
    chessboard->chessBB[src_no] ^= fromToBB;
    chessboard->colorBB[src_no / 7] ^= fromToBB;
    chessboard->emptyBB[src] = 1;
    chessboard->emptyBB[dst] = 0;
    chessboard->board[dst] = chessboard->board[src];
    chessboard->board[src] = CHESS_EMPTY;
    chessboard->hash ^= random_table[src_no][src] ^ random_table[src_no][dst];
    // if(chessboard->hash != zobrist(chessboard)) fprintf(stderr, "not
    // equal\n");
  }
}
void MyAI::generateMove(char move[6]) {
#ifdef WINDOWS
  begin = clock();
#else
  gettimeofday(&begin, 0);
#endif
  /*
    int debug_moves[128];
    int debug_moves_count = expand(this->main_chessboard, debug_moves,
    this->agent_color); for (int i = 0; i < debug_moves_count; i++) { int
    src_idx = debug_moves[i] << 5, dst_idx = debug_moves[i] & 31; if
    (main_chessboard.board[src_idx] % 7 == 1) { fprintf(stderr, "gun: %d ->
    %d\n", src_idx, dst_idx);
      }
    }
  */
  if (main_chessboard.colorBB[0].count() + main_chessboard.colorBB[1].count() +
          main_chessboard.emptyBB.count() >=
      8)
    time_limit = 12.;
  else if (900 - game_elapsed <= 300)
    time_limit = 5.;
  else
    time_limit = 9.;

  collision = hit = 0;
  best = search_cnt = 0;
  int best_move, best_move_tmp = 0, iterative_depth = 3;
  int update_best_move = -1;
  double score = 0.;
  double last_elapsed = 0., former_elapsed = 0.;
  std::vector<int> pv, pv_tmp;
  long seconds, microseconds;

  if (first_play) {  // opening strategy
    if (main_chessboard.board[5] == CHESS_COVER)
      best_move = (5 << 5) | 5;
    else
      best_move = (26 << 5) | 26;
    first_play = 0;
    goto statistics;
  }

  /*if (main_chessboard.coverBB.none() &&
      (main_chessboard.red_chess_num <= 5 ||
       main_chessboard.black_chess_num <= 5)) {
    best_move = MCS_pure();
    goto statistics;
  }*/

  // calculate iterative depth = 3
  memset(HT, 0, sizeof(HT));
  score = negaScout(this->main_chessboard, &best_move_tmp, this->agent_color, 0,
                    iterative_depth, -DBL_MAX, DBL_MAX, &pv_tmp, -100);
#ifdef WINDOWS
  clock_t end = clock();
  last_elapsed = (end - start_time);
#else
  struct timeval end;
  gettimeofday(&end, 0);
  seconds = end.tv_sec - begin.tv_sec;
  microseconds = end.tv_usec - begin.tv_usec;
  last_elapsed = (seconds * 1000 + microseconds * 1e-3);
#endif
  /*
  if (isTimeUp()) {
    // rule-based flip
    int op = agent_color ^ 1;
    int found = 0, x;
    std::bitset<32> b, bb;
    b = main_chessboard.chessBB[op + 6] | main_chessboard.chessBB[op + 3] |
        main_chessboard.chessBB[op + 2] | main_chessboard.chessBB[op + 1] |
        main_chessboard.chessBB[op + 0];
    while (!found && (x = popLSB(&b)) != -1) {
      bb = pmoves[x] & main_chessboard.coverBB;
      if ((x = popLSB(&b)) != -1) {
        best_move = (x << 5) | x;
        found = 1;
      }
    }
    if (!found) {
      int flip_moves[128];
      int flip_cnt = expandFlip(main_chessboard, flip_moves);
      int idx = rand() % flip_cnt;
      best_move = flip_moves[idx];
    }
  }
  */

  for (iterative_depth = 5; !isTimeUp(); iterative_depth += 2) {
    memset(HT, 0, sizeof(HT));

#ifdef WINDOWS
    clock_t start_time;
    start_time = clock();

    double remain_time = (start_time - begin);
#else
    struct timeval start_time;
    gettimeofday(&start_time, 0);
    seconds = start_time.tv_sec - begin.tv_sec;
    microseconds = start_time.tv_usec - begin.tv_usec;
    double remain_time = (seconds * 1000 + microseconds * 1e-3);
#endif

    pv = pv_tmp;
    best_move = best_move_tmp;
    update_best_move++;
    if (iterative_depth >= 6 &&
        time_limit * 1000 - remain_time <
            last_elapsed / former_elapsed *
                last_elapsed) {  // iterative_depth >= 6??
      break;
    }
    pv_tmp.clear();
    score = negaScout(this->main_chessboard, &best_move_tmp, this->agent_color,
                      0, iterative_depth, -DBL_MAX, DBL_MAX, &pv_tmp, -100);

    former_elapsed = last_elapsed;
#ifdef WINDOWS
    clock_t end = clock();
    last_elapsed = (end - start_time);
#else
    struct timeval end;
    gettimeofday(&end, 0);
    seconds = end.tv_sec - start_time.tv_sec;
    microseconds = end.tv_usec - start_time.tv_usec;
    last_elapsed = (seconds * 1000 + microseconds * 1e-3);
#endif
  }
  if (update_best_move <= 0) {
    best_move = best_move_tmp;  // with chance node, the it can not completely
                                // search depth = 4 QQ
    pv = pv_tmp;
  }

statistics:
  int start_point = best_move >> 5;
  int end_point = best_move & 31;
  sprintf(move, "%c%c-%c%c", 'a' + (start_point % 4),
          '1' + (7 - start_point / 4), 'a' + (end_point % 4),
          '1' + (7 - end_point / 4));

  fprintf(stderr, "iterative depth: %d, score = %lf\n", iterative_depth, score);
  fprintf(stderr, "hit = %d, collision = %d, collision probability = %lf\n",
          hit, collision, 1.0 * collision / (hit + collision));
  fprintf(stderr, "average best move occurs at %lf move\n",
          1.0 * best / search_cnt + 1);
  fprintf(stderr, "pv size = %d\n", pv.size());
  for (int i = pv.size() - 1; i >= 0; i--) {
    fprintf(stderr, "%d ", pv[i]);
  }
  fprintf(stderr, "\n");

#ifdef WINDOWS
  end = clock();
  game_elapsed += (end - begin);
#else
  gettimeofday(&end, 0);
  seconds = end.tv_sec - begin.tv_sec;
  microseconds = end.tv_usec - begin.tv_usec;
  game_elapsed += (seconds * 1000 + microseconds * 1e-3);
#endif
}
int tb_size = 22;
Entry transposition_table[2][1 << 22];
double MyAI::negaScout(ChessBoard chessboard, int* move, const int color,
                       const int depth, const int remain_depth, double alpha,
                       double beta, std::vector<int>* pv, int last_chance) {
  int moves[128];
  int move_count = expand(chessboard, moves, color);
  int flip_moves[64], flip_count;
  flip_count = expandFlip(chessboard, flip_moves);

  if (isTimeUp() || chessboard.red_chess_num == 0 ||
      chessboard.black_chess_num == 0 || move_count + flip_count == 0 ||
      isDraw(&chessboard) || remain_depth <= 0) {
    return evaluate(&chessboard, move_count + flip_count, color, depth) *
           (color == this->agent_color ? 1 : -1);
  }  // this should be placed before transposition table, in case of position
     // repetition draw.

  double m = -DBL_MAX;

  Entry entry = transposition_table[color][chessboard.hash];
  if (entry.exact != -1) {
    int identical = 1;
    for (int i = 0; i < 14; i++) {
      if (entry.chessBB[i] != chessboard.chessBB[i]) {
        identical = 0;
        collision++;
        break;
      }
    }
    if (identical) hit++;
    if (identical && entry.height >= remain_depth) {
      if (entry.exact == 1) {
        *move = entry.move;
        pv->push_back(*move);
        return entry.score;
      } else if (entry.exact == 2) {  // upperbound
        if (entry.score < beta) {
          beta = entry.score;
          *move = entry.move;
          pv->push_back(*move);
        }
        if (beta <= alpha) {
          return beta;
        }
      } else {  // lowerbound
        if (entry.score > alpha) {
          alpha = entry.score;
          *move = entry.move;
          pv->push_back(*move);
        }
        if (alpha >= beta) {
          return alpha;
        }
      }
    } else if (identical && entry.height < remain_depth) {
      moves[move_count++] = entry.move;  // TODO: do once
    }
  }

  double n = beta;
  int where = 0;
  for (int i = 0; i < move_count; i++) {
    ChessBoard new_chessboard = chessboard;
    makeMove(&new_chessboard, moves[i], 0);
    std::vector<int> pv_child;
    int best_move;
    double t = -negaScout(new_chessboard, &best_move, color ^ 1, depth + 1,
                          remain_depth - 1, -n, -std::max(alpha, m), &pv_child,
                          last_chance);
    if (t > m) {
      *move = moves[i];
      *pv = pv_child;
      pv->push_back(*move);
      where = i;
      if (abs(n - beta) < epsilon || remain_depth < 3 || t >= beta) {
        m = t;
      } else {
        m = -negaScout(new_chessboard, &best_move, color ^ 1, depth + 1,
                       remain_depth - 1, -beta, -t, pv, last_chance);
      }
    }

    if (m >= beta) {
      best += i;
      search_cnt++;
      HT[*move] += 1 << remain_depth;
      if (m <= 0.9)
        transposition_table[color][chessboard.hash] =
            Entry(chessboard.chessBB, chessboard.coverBB, m, remain_depth, 0,
                  moves[i]);
      return m;
    }

    n = std::max(alpha, m) + epsilon;
  }

  if (move_count == 0 || (remain_depth > 3 && depth - last_chance >= 3)) {
    for (int i = 0; i < flip_count; i++) {
      std::vector<int> pv_child;
      double t = star0(chessboard, flip_moves[i], color ^ 1, depth + 1,
                       remain_depth - 1, std::max(alpha, m), beta, &pv_child);
      if (t > m) {
        *move = flip_moves[i];
        *pv = pv_child;
        pv->push_back(*move);
        where = move_count + i;
        m = t;
      }
      if (m >= beta) {
        best += move_count + i;
        search_cnt++;
        HT[*move] += 1 << remain_depth;
        if (m <= 0.9)
          transposition_table[color][chessboard.hash] =
              Entry(chessboard.chessBB, chessboard.coverBB, m, remain_depth, 0,
                    flip_moves[i]);
      }
    }
  }

  if (m > alpha && m <= 0.9) {
    transposition_table[color][chessboard.hash] = Entry(
        chessboard.chessBB, chessboard.coverBB, m, remain_depth, 1, *move);
  } else if (m <= 0.9) {
    transposition_table[color][chessboard.hash] = Entry(
        chessboard.chessBB, chessboard.coverBB, m, remain_depth, 2, *move);
  }
  best += where;
  search_cnt++;
  HT[*move] += 1 << remain_depth;
  return m;
}
double MyAI::star0(ChessBoard chessboard, int move, int color, int depth,
                   int remain_depth, double alpha, double beta,
                   std::vector<int>* pv) {
  double total = 0;
  double remain_total = 0;

  for (int i = 0; i < 14; i++) {
    if (chessboard.cover_chess[i] == 0) continue;
    ChessBoard new_chessboard = chessboard;
    makeMove(&new_chessboard, move, i);
    int best_move;
    double t = -negaScout(new_chessboard, &best_move, color, depth,
                          remain_depth, -DBL_MAX, DBL_MAX, pv, depth - 1);
    total += chessboard.cover_chess[i] * t;
    remain_total += chessboard.cover_chess[i];
  }
  double E_score =
      (1.0 * total / remain_total);  // calculate the expect value of flip
  return E_score;
}
double MyAI::star1(ChessBoard chessboard, int move, int color, int depth,
                   int remain_depth, double alpha, double beta,
                   std::vector<int>* pv) {
  double P[14];
  double sum = 0.;
  for (int i = 0; i < 14; i++) {
    P[i] = chessboard.cover_chess[i];
    sum += chessboard.cover_chess[i];
  }
  for (int i = 0; i < 14; i++) P[i] /= sum * 1.0;

  double A, B, m, M, vexp;
  double vmax = 1., vmin = -1.;
  A = (alpha - vmax) / P[0] + vmax;
  B = (beta - vmin) / P[0] + vmin;
  m = vmin;
  M = vmax;
  vexp = 0.;
  for (int i = 0; i < 14; i++) {
    if (chessboard.cover_chess[i] == 0) continue;
    ChessBoard new_chessboard = chessboard;
    makeMove(&new_chessboard, move, i);
    int best_move;
    double t = negaScout(new_chessboard, &best_move, color, depth, remain_depth,
                         std::max(A, vmin), std::min(B, vmax), pv, depth - 1);
    m = m + P[i] * (t - vmin);
    M = M + P[i] * (t - vmax);
    if (t >= B) return m;
    if (t <= A) return M;
    vexp += P[i] * t;
    if (i < 13) A = (P[i] * A - P[i + 1] * vmax - P[i] * t) / P[i + 1];
    if (i < 13) B = (P[i] * B - P[i + 1] * vmin - P[i] * t) / P[i + 1];
  }
  return vexp;
}
double MyAI::alphaBeta(ChessBoard chessboard, int* move, const int color,
                       int depth, const int remain_depth, double alpha,
                       double beta) {
  int moves[128];
  int move_count = expand(chessboard, moves, color);

  if (isTimeUp() || chessboard.red_chess_num == 0 ||
      chessboard.black_chess_num == 0 || move_count == 0 ||
      isDraw(&chessboard) || remain_depth <= 0) {
    return evaluate(&chessboard, move_count, color, depth) *
           (color == this->agent_color ? 1 : -1);
  }

  double m = alpha;
  for (int i = 0; i < move_count; i++) {
    ChessBoard new_chessboard = chessboard;
    makeMove(&new_chessboard, moves[i], 0);
    int best_move = 0;
    double t = -alphaBeta(new_chessboard, &best_move, color ^ 1, depth + 1,
                          remain_depth - 1, -beta, -m);
    if (t > m) {
      m = t;
      *move = moves[i];
    }
    if (m >= beta) {
      return m;
    }
  }
  return m;
}
int MyAI::getIndex(std::bitset<32>* LSB) {
  return index32[((LSB->to_ullong() * 0x08ed2be6ull) >> 27ull) % 32ull];
}
int MyAI::popLSB(std::bitset<32>* BB) {
  if (BB->none()) return -1;
  std::bitset<32> LSB = *BB & std::bitset<32>(-BB->to_ulong());
  *BB ^= LSB;
  return getIndex(&LSB);
}
int MyAI::popMSB(std::bitset<32>* BB) {
  if (BB->none()) return -1;
  std::bitset<32> MSB = *BB;
  MSB |= MSB >> 16;
  MSB |= MSB >> 8;
  MSB |= MSB >> 4;
  MSB |= MSB >> 2;
  MSB |= MSB >> 1;
  MSB = std::bitset<32>((MSB >> 1).to_ulong() + 1);
  *BB ^= MSB;
  return getIndex(&MSB);
}
int MyAI::expandGun(ChessBoard chessboard, int* moves, int color) {
  if (color != 0 && color != 1) return 0;
  int move_count = 0;
  int chess_no = 1 + color * 7;
  int src_idx, dst_idx;
  std::bitset<32> dst(0), tmp(0);
  while ((src_idx = popLSB(&chessboard.chessBB[chess_no])) != -1) {
    dst = pmoves[src_idx] & chessboard.emptyBB;
    while ((dst_idx = popLSB(&dst)) != -1) {
      moves[move_count++] = (src_idx << 5) + dst_idx;
    }

    int r = src_idx / 4, c = src_idx % 4;
    // left
    tmp = (row_mask[r] & ~chessboard.emptyBB) << (4 * (7 - r) + 4 - c) >>
          (4 * (7 - r) + 4 - c);
    popMSB(&tmp);
    if ((dst_idx = popMSB(&tmp)) != -1 &&
        chessboard.colorBB[color ^ 1][dst_idx]) {
      moves[move_count++] = (src_idx << 5) + dst_idx;
    }

    // right
    tmp = (row_mask[r] & ~chessboard.emptyBB) >> (4 * r + c + 1)
                                                     << (4 * r + c + 1);
    popLSB(&tmp);
    if ((dst_idx = popLSB(&tmp)) != -1 &&
        chessboard.colorBB[color ^ 1][dst_idx]) {
      moves[move_count++] = (src_idx << 5) + dst_idx;
    }

    // up
    tmp = (column_mask[c] & ~chessboard.emptyBB) << (4 * (7 - r) + 4 - c) >>
          (4 * (7 - r) + 4 - c);
    popMSB(&tmp);
    if ((dst_idx = popMSB(&tmp)) != -1 &&
        chessboard.colorBB[color ^ 1][dst_idx]) {
      moves[move_count++] = (src_idx << 5) + dst_idx;
    }

    // down
    tmp = (column_mask[c] & ~chessboard.emptyBB) >> (4 * r + c + 1)
                                                        << (4 * r + c + 1);
    popLSB(&tmp);
    if ((dst_idx = popLSB(&tmp)) != -1 &&
        chessboard.colorBB[color ^ 1][dst_idx]) {
      moves[move_count++] = (src_idx << 5) + dst_idx;
    }
  }

  return move_count;
}

int MyAI::expand(ChessBoard chessboard, int* moves, int color) {
  if (color != 0 && color != 1) return 0;
  int move_count = expandGun(chessboard, moves, color);
  for (int chess_no = 0; chess_no < 14; chess_no++) {
    if (chess_no / 7 != color) continue;
    int src_idx, dst_idx;
    std::bitset<32> dst(0);
    while ((src_idx = popLSB(&chessboard.chessBB[chess_no])) != -1) {
      switch (chess_no) {
        case 0:
          dst = pmoves[src_idx] & (chessboard.emptyBB | chessboard.chessBB[7] |
                                   chessboard.chessBB[13]);
          break;
        case 2:
          dst =
              pmoves[src_idx] & (chessboard.emptyBB | chessboard.chessBB[7] |
                                 chessboard.chessBB[8] | chessboard.chessBB[9]);
          break;
        case 3:
          dst =
              pmoves[src_idx] & (chessboard.emptyBB | chessboard.chessBB[7] |
                                 chessboard.chessBB[8] | chessboard.chessBB[9] |
                                 chessboard.chessBB[10]);
          break;
        case 4:
          dst = pmoves[src_idx] &
                (chessboard.emptyBB |
                 (chessboard.colorBB[1] ^ chessboard.chessBB[13] ^
                  chessboard.chessBB[12]));
          break;
        case 5:
          dst = pmoves[src_idx] &
                (chessboard.emptyBB |
                 (chessboard.colorBB[1] ^ chessboard.chessBB[13]));
          break;
        case 6:
          dst = pmoves[src_idx] &
                (chessboard.emptyBB |
                 (chessboard.colorBB[1] ^ chessboard.chessBB[7]));
          break;
        case 7:
          dst = pmoves[src_idx] & (chessboard.emptyBB | chessboard.chessBB[0] |
                                   chessboard.chessBB[6]);
          break;
        case 9:
          dst =
              pmoves[src_idx] & (chessboard.emptyBB | chessboard.chessBB[0] |
                                 chessboard.chessBB[1] | chessboard.chessBB[2]);
          break;
        case 10:
          dst =
              pmoves[src_idx] & (chessboard.emptyBB | chessboard.chessBB[0] |
                                 chessboard.chessBB[1] | chessboard.chessBB[2] |
                                 chessboard.chessBB[3]);
          break;
        case 11:
          dst = pmoves[src_idx] &
                (chessboard.emptyBB |
                 (chessboard.colorBB[0] ^ chessboard.chessBB[6] ^
                  chessboard.chessBB[5]));
          break;
        case 12:
          dst = pmoves[src_idx] &
                (chessboard.emptyBB |
                 (chessboard.colorBB[0] ^ chessboard.chessBB[6]));
          break;
        case 13:
          dst = pmoves[src_idx] &
                (chessboard.emptyBB |
                 (chessboard.colorBB[0] ^ chessboard.chessBB[0]));
          break;
      }
      while ((dst_idx = popLSB(&dst)) != -1) {
        moves[move_count++] = (src_idx << 5) + dst_idx;
      }
    }
  }
  std::sort(moves, moves + move_count,
            [&](int x, int y) { return HT[x] > HT[y]; });
  return move_count;
}

int MyAI::expandFlip(ChessBoard chessboard, int* flip_moves) {
  int flip_count = 0;
  int x;
  while ((x = popLSB(&chessboard.coverBB)) != -1) {
    flip_moves[flip_count++] = (x << 5) + x;
  }
  return flip_count;
}

int MyAI::definitely_win(ChessBoard* chessboard) {
  auto totalpiece = [&](int id) {
    return chessboard->chessBB[id].count() + chessboard->cover_chess[id];
  };
  if (totalpiece(7 * agent_color + 6) &&
      totalpiece(7 * (agent_color ^ 1) + 0) == 0 &&
      totalpiece(7 * agent_color + 5) >= totalpiece(7 * (agent_color ^ 1) + 5))
    return 1;
  if (totalpiece(7 * (agent_color ^ 1) + 6) == 0 &&
      totalpiece(7 * agent_color + 5) == 2 &&
      totalpiece(7 * (agent_color ^ 1) + 5) <= 1)
    return 1;
  if (totalpiece(7 * (agent_color ^ 1) + 6) &&
      totalpiece(7 * agent_color + 0) == 0 &&
      totalpiece(7 * (agent_color ^ 1) + 5) >= totalpiece(7 * agent_color + 5))
    return -1;
  if (totalpiece(7 * agent_color + 6) == 0 &&
      totalpiece(7 * (agent_color ^ 1) + 5) == 2 &&
      totalpiece(7 * agent_color + 5) <= 1)
    return -1;
  return 0;
}

double MyAI::evaluate(ChessBoard* chessboard, int move_count, int color,
                      int depth) {
  // use my point of view

  double score = 0;
  if (move_count == 0 || isDraw(chessboard)) {
    if (color == this->agent_color || isDraw(chessboard))
      score += (LOSE - WIN) + (LOSE - WIN) / depth * 2;
    else
      score += (WIN - LOSE) + (WIN - LOSE) / depth * 2;
  } else {
    static double values[14] = {18, 180, 18, 36, 90, 270, 320,
                                18, 180, 18, 36, 90, 270, 320};

    double piece_value = 0.;
    for (int i = 0; i < 14; i++) {
      if (i / 7 == this->agent_color)
        piece_value += chessboard->chessBB[i].count() * values[i];
      else
        piece_value -= chessboard->chessBB[i].count() * values[i];
    }
    score = piece_value / 1478.;

    static double influence[14][14] = {
        {0, 0, 0, 0, 0, 0, 0, 0.042, 0, 0, 0, 0, 0, 0.45},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0.05, 0.2, 0.092, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0.05, 0.2, 0.1, 0.14, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0.05, 0.2, 0.1, 0.15, 0.24, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0.05, 0.2, 0.1, 0.15, 0.25, 0.44, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0.2, 0.1, 0.15, 0.25, 0.45, 0.5},
        {0.042, 0, 0, 0, 0, 0, 0.45, 0, 0, 0, 0, 0, 0, 0},
        {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0.05, 0.2, 0.092, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0.05, 0.2, 0.1, 0.14, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0.05, 0.2, 0.1, 0.15, 0.24, 0, 0, 0, 0, 0, 0, 0, 0, 0},
        {0.05, 0.2, 0.1, 0.15, 0.25, 0.44, 0, 0, 0, 0, 0, 0, 0, 0},
        {0, 0.2, 0.1, 0.15, 0.25, 0.45, 0.5, 0, 0, 0, 0, 0, 0, 0}};
    if (chessboard->red_chess_num <= 5 || chessboard->black_chess_num <= 5) {
      // manhattan distance
      int pos[2][16];
      int piece_cnt[2] = {0};
      for (int i = 0; i < 14; i++) {
        int x;
        for (std::bitset<32> b = chessboard->chessBB[i];
             (x = popLSB(&b)) != -1;) {
          pos[i / 7][piece_cnt[i / 7]++] = x;
        }
      }
      int real_influence = 0;
      for (int i = 0; i < piece_cnt[agent_color]; i++) {
        for (int j = 0; j < piece_cnt[agent_color ^ 1]; j++) {
          int from = pos[agent_color][i], to = pos[agent_color ^ 1][j];
          int md = abs(from / 4 - to / 4) + abs(from % 4 - to % 4);
          real_influence +=
              ((md == 1)
                   ? influence[chessboard->board[from]][chessboard->board[to]]
                   : influence[chessboard->board[from]][chessboard->board[to]] *
                         pow(1.5, 2 - md));
          real_influence -=
              ((md == 1)
                   ? influence[chessboard->board[to]][chessboard->board[from]] /
                         1.5
                   : influence[chessboard->board[to]][chessboard->board[from]] *
                         pow(1.5, 2 - md));
        }
      }
      if (piece_cnt[agent_color ^ 1] < 2)
        real_influence += (2 - piece_cnt[agent_color ^ 1]) * 0.45 * 1.5;
      if (piece_cnt[agent_color] < 2)
        real_influence -= (2 - piece_cnt[agent_color]) * 0.45 * 1.5;
      score = score * 7. / 8. + 1.0 * real_influence / 8.;

      /*
      // mobility
      double mobility = 0.;
      for (int i = 0; i < 14; i++) {
        int x;
        for (std::bitset<32> b = chessboard->chessBB[i];
             (x = popLSB(&b) != -1);) {
          if (i / 7 == agent_color)
            mobility -= 0.25 * (4 - pmoves[x].count()) * values[i];
          else
            mobility += 0.25 * (4 - pmoves[x].count()) * values[i];
        }
      }
      score += mobility / 1943. / 9.;
      */
    }
    /*
    if (chessboard->red_chess_num + chessboard->black_chess_num <= 5 ||
        chessboard->red_chess_num <= 3 || chessboard->black_chess_num <= 3) {
      int pos[2][2];
      int pos_cnt[2] = {0};
      double distance = 0;
      for (int k = 0; k < 2; k++) {
        pos_cnt[0] = pos_cnt[1] = 0;
        for (int i = k + 6; i >= k + 0 && pos_cnt[k] < 2; i++) {
          if (chessboard->chessBB[i].count()) {
            int x;
            for (std::bitset<32> BB = chessboard->chessBB[i];
                 pos_cnt[k] < 2 && (x = popLSB(&BB) != -1);) {
              pos[k][pos_cnt[k]++] = x;
            }
          }
        }
        for (int i = (k ^ 1) + 0; i < (k ^ 1) + 7 && pos_cnt[(k ^ 1)] < 2;
             i++) {
          if (chessboard->chessBB[i].count()) {
            int x;
            for (std::bitset<32> BB = chessboard->chessBB[i];
                 pos_cnt[k ^ 1] < 2 && (x = popLSB(&BB) != -1);) {
              pos[k ^ 1][pos_cnt[k ^ 1]++] = x;
            }
          }
        }
        double value = 0.;
        if (influence[chessboard->board[pos[k][0]]]
                     [chessboard->board[pos[k ^ 1][0]]] > 0.) {
          value += 0.1 * (12 - abs(pos[k][0] / 4 - pos[k ^ 1][0] / 4) -
                          abs(pos[k][0] % 4 - pos[k ^ 1][0] % 4));
        }
        if (influence[chessboard->board[pos[k][1]]]
                     [chessboard->board[pos[k ^ 1][0]]] > 0.) {
          value += 0.08 * (12 - abs(pos[k][1] / 4 - pos[k ^ 1][0] / 4) -
                           abs(pos[k][1] % 4 - pos[k ^ 1][0] % 4));
        }
        if (k == agent_color)
          distance += value;
        else
          distance -= value;
      }
      score = score * 0.7 + distance * 0.3;
    }
    */

    // definitely win / lose
    if (definitely_win(chessboard) == 1)
      score = score * 0.2 + 0.8;
    else if (definitely_win(chessboard) == -1)
      score = score * 0.2 - 0.8;

    score = score * (WIN - 0.01);
  }
  return score;
}
void MyAI::print_chessboard(ChessBoard* chessboard, int color) {
  char Mes[1024] = "";
  char temp[1024];
  char myColor[10] = "";
  if (color == -99)
    strcpy(myColor, "Unknown");
  else if (color == RED)
    strcpy(myColor, "Red");
  else
    strcpy(myColor, "Black");
  sprintf(temp, "------------%s-------------\n", myColor);
  strcat(Mes, temp);
  strcat(Mes, "<8> ");
  for (int i = 0; i < 32; i++) {
    if (i != 0 && i % 4 == 0) {
      sprintf(temp, "\n<%d> ", 8 - (i / 4));
      strcat(Mes, temp);
    }
    char chess_name[4] = "";
    sprintf(chess_name, " %c ", skind[chessboard->board[i]]);
    sprintf(temp, "%5s", chess_name);
    strcat(Mes, temp);
  }
  strcat(Mes, "\n\n     ");
  for (int i = 0; i < 4; i++) {
    sprintf(temp, " <%c> ", 'a' + i);
    strcat(Mes, temp);
  }
  strcat(Mes, "\n\n");
  printf("%s", Mes);
}

bool MyAI::isDraw(const ChessBoard* chessboard) {
  // No Eat Flip
  if (chessboard->no_eat_flip >= NOEATFLIP_LIMIT) {
    return true;
  }

  // Position Repetition
  int last_idx = chessboard->history_count - 1;
  // -2: my previous ply
  int idx = last_idx - 2;
  // All ply must be move type
  int smallest_repetition_idx =
      last_idx - (chessboard->no_eat_flip / POSITION_REPETITION_LIMIT);
  // check loop
  while (idx >= 0 && idx >= smallest_repetition_idx) {
    if (chessboard->history[idx] == chessboard->history[last_idx]) {
      // how much ply compose one repetition
      int repetition_size = last_idx - idx;
      bool isEqual = true;
      for (int i = 1; i < POSITION_REPETITION_LIMIT && isEqual; ++i) {
        for (int j = 0; j < repetition_size; ++j) {
          int src_idx = last_idx - j;
          int checked_idx = last_idx - i * repetition_size - j;
          if (chessboard->history[src_idx] !=
              chessboard->history[checked_idx]) {
            isEqual = false;
            break;
          }
        }
      }
      if (isEqual) {
        return true;
      }
    }
    idx -= 2;
  }

  return false;
}
bool MyAI::isTimeUp() {
  double elapsed;  // ms

  // design for different os
#ifdef WINDOWS
  clock_t end = clock();
  elapsed = (end - begin);
#else
  struct timeval end;
  gettimeofday(&end, 0);
  long seconds = end.tv_sec - begin.tv_sec;
  long microseconds = end.tv_usec - begin.tv_usec;
  elapsed = (seconds * 1000 + microseconds * 1e-3);
#endif

  return elapsed >= time_limit * 1000;
}

/******************** MCS pure from baseline **********************/

// always use my point of view, so use this->Color
double MyAI::MCS_evaluate(const ChessBoard* chessboard,
                          const int legal_move_count, const int color) {
  // score = My Score - Opponent's Score
  double score = 0;

  if (legal_move_count == 0) {   // Win, Lose
    if (color == agent_color) {  // Lose
      score += LOSE - WIN;
    } else {  // Win
      score += WIN - LOSE;
    }
  } else if (isDraw(chessboard)) {  // Draw
                                    // score = DRAW - DRAW;
  }

  // Bonus (Only Win / Draw)
  // static material values
  // empty is zero
  const double values[14] = {1, 180, 6, 18, 90, 270, 810,
                             1, 180, 6, 18, 90, 270, 810};

  double piece_value = 0;
  // flipped
  for (int i = 0; i < 32; i++) {
    if (chessboard->board[i] != CHESS_EMPTY &&
        chessboard->board[i] != CHESS_COVER) {
      if (chessboard->board[i] / 7 == agent_color) {
        piece_value += values[chessboard->board[i]];
      } else {
        piece_value -= values[chessboard->board[i]];
      }
    }
  }
  // covered
  for (int i = 0; i < 14; ++i) {
    if (chessboard->cover_chess[i] > 0) {
      if (i / 7 == agent_color) {
        piece_value += chessboard->cover_chess[i] * values[i];
      } else {
        piece_value -= chessboard->cover_chess[i] * values[i];
      }
    }
  }

  if (legal_move_count == 0 && color == agent_color) {  // I lose
    if (piece_value > 0) {                              // but net value > 0
      piece_value = 0;
    }
  } else if (legal_move_count == 0 && color != agent_color) {  // Opponent lose
    if (piece_value < 0) {  // but net value < 0
      piece_value = 0;
    }
  }

  // linear map to [-<BONUS>, <BONUS>]
  // score max value = 1*5 + 180*2 + 6*2 + 18*2 + 90*2 + 270*2 + 810*1 = 1943
  // <ORIGINAL_SCORE> / <ORIGINAL_SCORE_MAX_VALUE> * <BONUS>
  piece_value = piece_value / 1943 * BONUS;
  score += piece_value;

  return score;
}

#define SIMULATE_COUNT_PER_CHILD 10
int MyAI::MCS_pure() {
  // Expand
  int moves[512];
  int move_count = expand(main_chessboard, moves, agent_color);

  // Create children
  ChessBoard* Children = new ChessBoard[move_count];
  double* Children_Scores = new double[move_count];
  for (int i = 0; i < move_count; ++i) {
    Children[i] = main_chessboard;
    makeMove(&Children[i], moves[i], 0);  // 0: dummy
    Children_Scores[i] = 0;               // reset
  }

  // MCS_pure
  int total_simulate_count = 0;
  while (!isTimeUp()) {
    // simulate every child <SIMULATE_COUNT_PER_CHILD> times
    for (int i = 0; i < move_count; ++i) {
      double total_score = 0;
      for (int k = 0; k < SIMULATE_COUNT_PER_CHILD; ++k) {
        total_score += simulate(Children[i]);
      }
      Children_Scores[i] += total_score;
    }
    total_simulate_count += SIMULATE_COUNT_PER_CHILD;
  }

  /*std::sort(moves, moves + move_count, [&](int x, int y) {
    return Children_Scores[x] > Children_Scores[y];
  });
  std::sort(Children_Scores, Children_Scores + move_count,
  std::greater<int>());*/

  for (int i = 0; i < move_count; ++i) {
    for (int j = i + 1; j < move_count; ++j) {
      if (Children_Scores[i] < Children_Scores[j]) {
        // swap
        double tmp_score = Children_Scores[i];
        Children_Scores[i] = Children_Scores[j];
        Children_Scores[j] = tmp_score;
        int tmp_move = moves[i];
        moves[i] = moves[j];
        moves[j] = tmp_move;
      }
    }
  }

  for (int i = 0; i < move_count; ++i) {
    char tmp[6];
    int tmp_start = moves[i] >> 5;
    int tmp_end = moves[i] & 31;
    // change int to char
    sprintf(tmp, "%c%c-%c%c", 'a' + (tmp_start % 4), '1' + (7 - tmp_start / 4),
            'a' + (tmp_end % 4), '1' + (7 - tmp_end / 4));

    fprintf(stderr, "%2d. Move: %s, Score: %+5lf, Sim_Count: %7d\n", i + 1, tmp,
            Children_Scores[i] / total_simulate_count, total_simulate_count);
    fflush(stderr);
  }

  return moves[0];
}

bool MyAI::isFinish(const ChessBoard* chessboard, int move_count) {
  return (chessboard->red_chess_num == 0 ||    // terminal node (no chess type)
          chessboard->black_chess_num == 0 ||  // terminal node (no chess type)
          move_count == 0 ||                   // terminal node (no move type)
          isDraw(chessboard)                   // draw
  );
}

double MyAI::simulate(ChessBoard chessboard) {
  int Moves[128];
  int moveNum;
  int turn_color = agent_color ^ 1;

  while (true) {
    // Expand
    moveNum = expand(chessboard, Moves, turn_color);

    // Check if is finish
    if (isFinish(&chessboard, moveNum)) {
      return MCS_evaluate(&chessboard, moveNum, turn_color);
    }

    // distinguish eat-move and pure-move
    int eatMove[128], eatMoveNum = 0;
    for (int i = 0; i < moveNum; ++i) {
      int dstPiece = chessboard.board[Moves[i] & 31];
      if (dstPiece != CHESS_EMPTY) {
        // eat-move
        eatMove[eatMoveNum] = Moves[i];
        eatMoveNum++;
      }
    }

    // Random Move
    bool selectEat = (eatMoveNum == 0 ? false : rand() % 2);
    int move =
        (selectEat ? eatMove[rand() % eatMoveNum] : Moves[rand() % moveNum]);
    makeMove(&chessboard, move, 0);  // 0: dummy

    // Change color
    turn_color ^= 1;
  }
}