for idx in tqdm(range(len(list_data_dict))):
        sample = list_data_dict[idx]
        ###MCx
        scores_true = []
        scores_false = []

        token_ranges=None
        input_text_keys = None
        context = sample['prefix']
        answer_true = ' ' + sample['completion']
        answers_false = []
        for i in range(3):
            answers_false.append(' ' + sample[f'contradiction_{i}'])
        
        generate_kwargs = dict(do_sample=args.do_sample, mode=args.decoding_method, mature_layer=mature_layer, candidate_premature_layers=candidate_premature_layers, relative_top=args.relative_top, relative_top_value=args.relative_top_value,evolution_rate=args.evolution_rate,evolution_scale=args.evolution_scale)
        completion_len = []
        answer_true_log_prob,correct_len,_= llm.lm_score(model_name_input,context, answer_true,single=single, start_layer=start_layer, end_layer=end_layer, attn_alpha=attn_alpha, token_enhance=token_enhance, token_weaken=token_weaken, beta=beta, sink=sink,sink_layers=sink_layers,ema=ema,th=th, **generate_kwargs)
        completion_len.append(correct_len)
        #answer_true_log_prob , _= llm.lm_score(model_name_input,context, answer_true,single=single, start_layer=start_layer, end_layer=end_layer, attn_alpha=attn_alpha, token_enhance=token_enhance, token_weaken=token_weaken, beta=beta, sink=sink,sink_layers=sink_layers,ema=ema,th=th, **generate_kwargs)
        ###MC
        scores_true.append(answer_true_log_prob)

        answer_false_log_probs = []
        for answer_false in answers_false:
            answer_false_log_prob ,incorrect_len,_= llm.lm_score(model_name_input,context, answer_false, single=single,start_layer=start_layer, end_layer=end_layer, attn_alpha=attn_alpha, token_enhance=token_enhance, token_weaken=token_weaken, beta=beta,sink=sink,sink_layers=sink_layers,ema=ema,th=th, **generate_kwargs)
            completion_len.append(incorrect_len)
            #answer_false_log_prob , _= llm.lm_score(model_name_input,context, answer_false, single=single,start_layer=start_layer, end_layer=end_layer, attn_alpha=attn_alpha, token_enhance=token_enhance, token_weaken=token_weaken, beta=beta,sink=sink,sink_layers=sink_layers,ema=ema,th=th, **generate_kwargs)
            answer_false_log_probs.append(answer_false_log_prob)
            ###MC
            scores_false.append(answer_false_log_prob)
       
        
        is_cor = True
        log_probs = [answer_true_log_prob] + answer_false_log_probs 
        normalized_log_probs = log_probs / np.array(completion_len)
        #normalized_log_probs = log_probs
        predicted_answer_idx = np.argmax(normalized_log_probs)
        if predicted_answer_idx == 0: 
            is_cor = True
        else:
            is_cor = False
        
        answers.append(is_cor)
        result_dict['is_correct'].append(is_cor)




for sample in tqdm(list_data_dict):
            # reference answers
            ref_best = format_best(sample['answer_best'])
            ref_true = split_multi_answer(sample['answer_true'])
            ref_false = split_multi_answer(sample['answer_false'])

            scores_true = []
            scores_false = []

            generate_kwargs = dict(mode=args.decoding_method, mature_layer=mature_layer,
                                   candidate_premature_layers=candidate_premature_layers,
                                   relative_top=args.relative_top, relative_top_value=args.relative_top_value,
                                   post_softmax=args.post_softmax,evolution_rate=args.evolution_rate,evolution_scale=args.evolution_scale)


            for temp_ans in ref_true:
                prompt, answer, demo, question = build_prompt_and_answer(sample['question'], temp_ans)
                # print('true')
                log_probs, c_dist = llm.lm_score(model_name_input,prompt, answer, demo, question, single=single,start_layer=start_layer, end_layer=end_layer, attn_alpha=attn_alpha, token_enhance=token_enhance, token_weaken=token_weaken, beta=beta,sink=sink,sink_layers=sink_layers,ema=ema,th=th,**generate_kwargs)
                scores_true.append(log_probs)


            for temp_ans in ref_false:
                prompt, answer, demo, question = build_prompt_and_answer(sample['question'], temp_ans)
                #print('true', answer)
                log_probs, c_dist = llm.lm_score(model_name_input,prompt, answer, demo, question, single=single,start_layer=start_layer, end_layer=end_layer, attn_alpha=attn_alpha, token_enhance=token_enhance, token_weaken=token_weaken, beta=beta,sink=sink,sink_layers=sink_layers,ema=ema,th=th,**generate_kwargs)
                scores_false.append(log_probs)


            #print('output',scores_true, scores_false)
            scores = MC_calcs(scores_true, scores_false, ref_true, ref_best)
            #print('score', scores)
            # check nan in mc1/2/3
            if np.isnan(scores['MC1']) or np.isnan(scores['MC2']) or np.isnan(scores['MC3']):
                import ipdb;

                ipdb.set_trace()

            result_dict['model_scores'].append(scores)
            result_dict['question'].append(sample)
            # update total scores
            result_dict['total_mc1'] += scores['MC1']
            result_dict['total_mc2'] += scores['MC2']
            result_dict['total_mc3'] += scores['MC3']
