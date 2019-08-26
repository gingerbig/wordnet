#include "forward_ui.h"
#include "load_data.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char dictionary[250][20]; // Ordered vocabulary

static void loadDictionary() {
  FILE *fpDict = fopen("data/vocab_ordered.txt", "r");
  int count;

  for (count = 0; fgets(dictionary[count], 20, fpDict); ++count) {
    dictionary[count][strlen(dictionary[count]) - 1] = '\0';
  }

  fclose(fpDict);
}

static void showDictionary() {
  int i;
  for (i = 0; i < 250; ++i) {
    printf("%s ", dictionary[i]);
  }
  printf("\n");
}

static int indexOfWord(const char *word) {
  int index = -1, i;
  for (i = 1; word && i < 251; ++i) {
    if (!strcmp(word, g_vocabulary[i])) {
      index = i;
      break;
    }
  }
  return index;
}

static int findMax(const WordNet *model) {
  int i, iMax = 0;
  double max = model->outputStateBatch[0];
  for (i = 1; i < model->vocabSize; ++i) {
    if (model->outputStateBatch[i] > max) {
      max = model->outputStateBatch[i];
      iMax = i;
    }
  }
  return iMax;
}

static char paragraphBuff[102400] = {0};

typedef struct _IndexedValue {
  int index;
  double value;
} IndexedValue;

static int compare(const void *p1, const void *p2) {
  if (((IndexedValue *)p1)->value > ((IndexedValue *)p2)->value) {
    return 1;
  } else {
    return -1;
  }
}

IndexedValue indexedOutputBuff[250];

void forward_ui(WordNet *model) {
  char line[1024];
  int indices[3];
  int currentIndex;
  int i;
  char delim[] = " ";
  char *p;

  loadDictionary();

  printf("##------Interactive UI------##\n");

  showDictionary();
  printf("|Input first 3 words > ");
  fgets(line, 1024, stdin);
  line[strlen(line) - 1] = '\0';
  p = strtok(line, delim);
  strcat(paragraphBuff, p);
  strcat(paragraphBuff, " ");
  indices[0] = indexOfWord(p);

  p = strtok(NULL, delim);
  strcat(paragraphBuff, p);
  strcat(paragraphBuff, " ");
  indices[1] = indexOfWord(p);

  p = strtok(NULL, delim);
  strcat(paragraphBuff, p);
  strcat(paragraphBuff, " ");
  indices[2] = indexOfWord(p);

  do {
    forwardPropagate(model, indices, 1);
    for (i = 0; i < 250; ++i) {
      indexedOutputBuff[i].index = i + 1;
      indexedOutputBuff[i].value = model->outputStateBatch[i];
    }
    qsort(indexedOutputBuff, 250, sizeof(IndexedValue), compare);

    puts(paragraphBuff);

    // Show top 5
    printf("*Top 5 = ");
    for (i = 0; i < 5; ++i) {
      printf("%d.%s(%lf) ", i + 1,
             g_vocabulary[indexedOutputBuff[249 - i].index],
             indexedOutputBuff[249 - i].value);
    }

    indices[0] = indices[1];
    indices[1] = indices[2];

    printf("\n|Choose a number (default = 1)> ");
    if (!fgets(line, 1024, stdin)) {
      return;
    }
    if ('1' <= line[0] && line[0] <= '5') {
      currentIndex = indexedOutputBuff[250 - (line[0] - '0')].index;
    } else {
      currentIndex = indexedOutputBuff[249].index;
    }

    indices[2] = currentIndex;
    strcat(paragraphBuff, g_vocabulary[currentIndex]);
    strcat(paragraphBuff, " ");
  } while (1);
}
