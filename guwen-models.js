/* eslint-disable no-console */
/* eslint-disable no-param-reassign */
/* eslint-disable no-use-before-define */
Vue.use(Toasted, {
  theme: 'bubble',
  position: 'bottom-right',
  duration: 5000
});

const api = axios.create({
  baseURL: 'https://api-inference.huggingface.co/models/ethanyt/',
  timeout: 10000
  // headers: { Authorization: 'Bearer api_RVKbgUeAOpyVomPVptdsquvBKASuzSuges' }
});

function delay(t, v) {
  return new Promise(((resolve) => {
    setTimeout(resolve.bind(null, v), t);
  }));
}

function apiRetry(url, data) {
  app.networkStatus = '';
  const f = () => api.post(url, data).catch((err) => {
    if (err.response && err.response.data.error.indexOf('is currently loading') !== -1) {
      app.networkStatus = '模型加载中';
      return delay(10000).then(f);
    }
    if (err.code === 'ECONNABORTED') {
      app.networkStatus = '请求超时 正在重试';
      return delay(1000).then(f);
    }
    if (err.response && err.response.data.error.indexOf('Rate limit') !== -1) {
      app.step = -1;
      Vue.toasted.show('当前已达使用上限 请稍后再试');
      app.networkStatus = '';
      return Promise.reject(new Error('ignore'));
    }

    if (err.response) console.log(err.response);
    app.step = -1;
    Vue.toasted.show('出错啦！再试一次');
    app.networkStatus = '';
    return Promise.reject(new Error('ignore'));
  });
  return f();
}

function runPunc(inputs) {
  return apiRetry('guwen-punc', { inputs });
}

function runQuote(inputs) {
  return apiRetry('guwen-quote', { inputs });
}

function runNer(inputs) {
  return apiRetry('guwen-ner', { inputs });
}

const puncMap = {
  ',': '，',
  '.': '。',
  '?': '？',
  '!': '！',
  '\\': '、',
  ':': '：',
  ';': '；'
};

function mergeMarks(arr1, arr2) {
  const merged = [];
  let cur = 0;
  arr2.forEach((item) => {
    for (; cur < arr1.length && arr1[cur].start <= item.start; cur += 1) {
      merged.push(arr1[cur]);
    }
    merged.push(item);
  });

  for (; cur < arr1.length; cur += 1) {
    merged.push(arr1[cur]);
  }

  let first = 0;
  for (let i = 0; i < merged.length; i += 1) {
    if (merged[first].start === merged[i].start) {
      if (merged[i].content === '」' && (merged[first].content === '，' || merged[first].content === '、' || merged[first].content === '：')) {
        [merged[i], merged[first]] = [merged[first], merged[i]];
      }
      if (merged[i].content === '》') {
        merged.splice(first, 0, merged.splice(i, 1)[0]);
      }
    } else {
      first = i;
    }
  }
  return merged;
}

function renderText(text, marks) {
  let html = '';
  let plain = '';
  let start = 0;
  marks.forEach((item) => {
    const s = text.slice(start, item.start);
    html += s;
    plain += s;
    start = item.start;
    html += `<span class="${item.type}">${item.content}</span>`;
    plain += item.content;
    if (item.type === 'ne') {
      start += item.content.length;
    }
  });
  const s = text.slice(start, text.length);
  html += s;
  plain += s;
  return {
    html, plain
  };
}

function decodeQuote(data) {
  // A simple greedy strategy
  const result = [];
  let entity = null;
  data.forEach((item) => {
    if (item.entity_group === 'B') {
      if (entity != null) result.push(entity);
      entity = { start: item.start, end: item.end };
    } else if (entity != null) entity.end = item.end;
  });
  if (entity != null) result.push(entity);
  return result;
}

const converter = OpenCC.Converter({ from: 'hk', to: 'cn' });

const examples = ['十年春齐师伐我公将战曹刿请见其乡人曰肉食者谋之又何间焉刿曰肉食者鄙未能远谋乃入见问何以战公曰衣食所安弗敢专也必以分人对曰小惠未徧民弗从也公曰牺牲玉帛弗敢加也必以信对曰小信未孚神弗福也公曰小大之狱虽不能察必以情对曰忠之属也可以一战战则请从',
  '秦王饮酒酣曰寡人窃闻赵王好音请奏瑟赵王鼓瑟秦御史前书曰某年月日秦王与赵王会饮令赵王鼓瑟蔺相如前曰赵王窃闻秦王善为秦声请奏盆缻秦王以相娱乐秦王怒不许于是相如前进缻因跪请秦王秦王不肯击缻相如曰五步之内相如请得以颈血溅大王矣左右欲刃相如相如张目叱之左右皆靡于是秦王不怿为一击缻相如顾召赵御史书曰某年月日秦王为赵王击缻秦之群臣曰请以赵十五城为秦王寿蔺相如亦曰请以秦之咸阳为赵王寿秦王竟酒终不能加胜于赵赵亦盛设兵以待秦秦不敢动',
  '夫人之相与俯仰一世或取诸怀抱晤言一室之内或因寄所托放浪形骸之外虽趣舍万殊静躁不同当其欣于所遇暂得于己快然自足曾不知老之将至及其所之既倦情随事迁感慨系之矣向之所欣俯仰之间已为陈迹犹不能不以之兴怀况修短随化终期于尽古人云死生亦大矣岂不痛哉',
  '臣密言臣以险衅夙遭闵凶生孩六月慈父见背行年四岁舅夺母志祖母刘悯臣孤弱躬亲抚养臣少多疾病九岁不行零丁孤苦至于成立既无伯叔终鲜兄弟门衰祚薄晚有儿息外无期功强近之亲内无应门五尺之僮茕茕孑立形影相吊而刘夙婴疾病常在床蓐臣侍汤药未曾废离'];

const app = new Vue({
  el: '#app',
  data: {
    stepName: '加注标点',
    step: -1,
    totalStep: 3,
    inputText: '',
    networkStatus: '',
    outputHtml: '点击下方按钮开始处理',
    exampleIdx: -1,
  },
  computed: {
    progress() {
      return `${Math.round((this.step / this.totalStep) * 100)}%`;
    },
    barText() {
      const networkStatus = this.networkStatus ? ` （${this.networkStatus}）` : '';
      return `${this.stepName}${networkStatus} ${this.step}/${this.totalStep}`;
    }
  },
  mounted() {
    this.change();
  },
  methods: {
    start() {
      this.step = 1;
      this.stepName = '加注标点';
      let text = this.inputText.replace(/[^\u4e00-\u9fff]/g, '');
      text = converter(text);
      let nerInput = '';
      let allMarks = [];
      runPunc(text)
        .then((res) => {
          allMarks = res.data
            .map((item) => ({ type: 'punc', content: puncMap[item.entity_group], start: item.end }));
          this.outputHtml = renderText(text, allMarks).html;
        })
        .then(() => delay(1000))
        .then(() => {
          this.step = 2;
          this.stepName = '加注引号';
          return runQuote(text);
        })
        .then((res) => {
          const quotes = [];
          decodeQuote(res.data).forEach((item) => {
            quotes.push({ type: 'quote', content: '「', start: item.start });
            quotes.push({ type: 'quote', content: '」', start: item.end });
          });
          allMarks = mergeMarks(allMarks, quotes);
          const { html, plain } = renderText(text, allMarks);
          this.outputHtml = html;
          nerInput = plain;
        })
        .then(() => delay(1000))
        .then(() => {
          this.step = 3;
          this.stepName = '实体识别';
          return runNer(nerInput);
        })
        .then((res) => {
          const entities = [];
          const books = [];
          let cur = 0;
          let offset = 0;
          res.data.forEach((item) => {
            while (cur < allMarks.length && allMarks[cur].start <= item.start - offset) {
              offset += 1;
              cur += 1;
            }
            const content = item.word.replace(/\s/g, '');
            if (content.match(/[^\u4e00-\u9fff]/)) return;
            if (item.entity_group === 'NOUN_OTHER') {
              entities.push({
                type: 'ne', content, start: item.start - offset, end: item.end - offset
              });
            } else {
              books.push({ type: 'book', content: '《', start: item.start - offset });
              books.push({ type: 'book', content: '》', start: item.end - offset });
            }
          });
          allMarks = mergeMarks(allMarks, books);
          allMarks = mergeMarks(allMarks, entities);
          const { html } = renderText(text, allMarks);
          this.outputHtml = html;
        })
        .then(() => {
          app.step = -1;
          app.networkStatus = '';
          Vue.toasted.show('处理完毕');
        })
        .catch((err) => {
          app.step = -1;
          app.networkStatus = '';
          if (err.message !== 'ignore') Vue.toasted.show('内部错误 请联系开发者');
        });
    },
    change() {
      if (this.exampleIdx === -1) {
        this.exampleIdx = Math.floor(Math.random() * examples.length);
      }
      this.inputText = examples[this.exampleIdx];
      this.exampleIdx += 1;
      if (this.exampleIdx >= examples.length) this.exampleIdx = 0;
    }
  }
});
