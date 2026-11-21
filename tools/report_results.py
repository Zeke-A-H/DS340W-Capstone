"""Export current measured results as CSV, LaTeX and paper-style PNG tables.

No model inference or metric recomputation. Missing measurements remain missing.
"""
import argparse
import csv
import json
import os
import re
from pathlib import Path
import textwrap

ROOT=Path(__file__).resolve().parents[1]
CONDITIONS=[('1_objects','1-object scenes'),('2_objects','2-object scenes'),
            ('3_objects','3-object scenes'),('4_objects','4-object scenes'),('all_objects','All scene sizes')]
CATEGORIES=[('chair','Chair'),('table','Table'),('storagefurniture','Cabinet / storage')]


def number(value, percent=False):
    return '—' if value is None else f'{value*100:.2f}' if percent else f'{value:.4f}'


def csv_file(path, rows, fields):
    with path.open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields)
        writer.writeheader();writer.writerows(rows)


def latex_file(path,headers,rows,caption):
    def escape(value):
        return str(value).replace('—','--').replace('%',r'\%').replace('_',r'\_').replace('&',r'\&').replace('↑',r'$\uparrow$').replace('↓',r'$\downarrow$').replace('\n',' ')
    text=['% Requires booktabs','\\begin{table}[ht]','\\centering',f'\\caption{{{escape(caption)}}}',
          '\\begin{tabular}{l'+'r'*(len(headers)-1)+'}',r'\toprule',
          ' & '.join(escape(h) for h in headers)+r' \\',r'\midrule']
    text+=[' & '.join(escape(v) for v in row)+r' \\' for row in rows]
    text += [r'\bottomrule',r'\end{tabular}',r'\end{table}']
    path.write_text('\n'.join(text)+'\n')


def figure_file(path,headers,rows,title,notes,groups=None):
    os.environ.setdefault('MPLCONFIGDIR',str(ROOT/'data/.cache/matplotlib'))
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','font.size':10})
    width=15 if len(headers)>8 else 11
    height=2.6+.36*len(rows)
    fig,ax=plt.subplots(figsize=(width,height));ax.axis('off')
    first=.25 if len(headers)>8 else .30
    widths=[first]+[(1-first)/(len(headers)-1)]*(len(headers)-1)
    table=ax.table(cellText=rows,colLabels=headers,colWidths=widths,cellLoc='center',bbox=[0,.24,1,.56])
    table.auto_set_font_size(False);table.set_fontsize(10)
    for (row,column),cell in table.get_celld().items():
        cell.set_facecolor('white');cell.visible_edges=''
        if row==0:cell.visible_edges='BT';cell.set_text_props(weight='bold')
        if row==len(rows):cell.visible_edges='B'
        cell.set_linewidth(.7)
        if column==0:cell.set_text_props(ha='left')
    if groups:
        import numpy as np
        positions=np.cumsum([0]+widths)
        for label,start,end in groups:
            ax.text((positions[start]+positions[end+1])/2,.86,label,ha='center',transform=ax.transAxes,fontsize=12)
            ax.plot([positions[start]+.005,positions[end+1]-.005],[.835,.835],color='black',linewidth=.6,transform=ax.transAxes)
    ax.set_title(title,fontsize=13,pad=28)
    ax.text(0,.13,'\n'.join(textwrap.wrap(notes,170 if len(headers)>8 else 125)),va='top',transform=ax.transAxes,fontsize=8.5)
    fig.savefig(path,dpi=220,bbox_inches='tight',facecolor='white');plt.close(fig)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output_dir',type=Path,default=ROOT/'data/output/comparisons')
    args=parser.parse_args();output=args.output_dir;output.mkdir(parents=True,exist_ok=True)
    records=[];size_rows=[];category_table=[];sources=[];status=[]
    for group,label in CONDITIONS:
        path=ROOT/'data/output/trassembler_h5/scene'/group/'test/metrics.json'
        if not path.is_file():
            status.append(dict(condition=label,status='missing metrics',path=str(path)));continue
        report=json.loads(path.read_text());sources.append(str(path));metrics=report['overall']
        size_rows.append(dict(condition=label,test_objects=metrics['total_samples'],cd=metrics.get('mean_cd'),
                              seg_accuracy_pct=None if metrics.get('partseg_accuracy') is None else metrics['partseg_accuracy']*100,
                              seg_miou_pct=None if metrics.get('partseg_instance_avg_iou') is None else metrics['partseg_instance_avg_iou']*100,
                              segmentation_objects=metrics.get('partseg_sample_count',0),
                              generation_success_pct=metrics['mean_occ']*100))
        row=[label]
        for category,display in CATEGORIES:
            entry=report.get('by_category',{}).get(category,{})
            record=dict(condition=label,category=category,test_objects=entry.get('total_samples',0),
                        cd=entry.get('mean_cd'),cd_objects=entry.get('valid_cd_count',0),
                        seg_accuracy_pct=None if entry.get('partseg_accuracy') is None else entry['partseg_accuracy']*100,
                        seg_miou_pct=None if entry.get('partseg_instance_avg_iou') is None else entry['partseg_instance_avg_iou']*100,
                        segmentation_objects=entry.get('partseg_sample_count',0),source=str(path))
            records.append(record)
            row += [number(entry.get('mean_cd')),number(entry.get('partseg_accuracy'),True),number(entry.get('partseg_instance_avg_iou'),True)]
        category_table.append(row)
    # Explicitly distinguish original isolated-object tests from category slices of scene crops.
    for category,_ in CATEGORIES:
        folder=ROOT/'data/output/trassembler_h5'/category
        status.append(dict(condition=f'Original isolated {category}',status='no reconstruction outputs' if not folder.exists() else 'check legacy evaluator logs; not included without saved metrics',path=str(folder)))
    placement={}
    for family in ('llamaft','trassembler'):
        checkpoint=ROOT/f'data/ckpts/{family}_placement'
        prediction=ROOT/f'data/output/{family}_placement'
        placement[family]=dict(checkpoints_found=checkpoint.exists() and any(checkpoint.rglob('*.safetensors' if family=='llamaft' else '*.ckpt')),
                               prediction_manifests_found=prediction.exists() and any(prediction.rglob('predictions.jsonl')))
    csv_file(output/'category_by_scene_condition.csv',records,list(records[0]))
    category_headers=['Training/test condition']+[name for _,display in CATEGORIES for name in (display+' CD ↓',display+' Seg Acc (%) ↑',display+' Seg mIoU (%) ↑')]
    missing=['Original isolated tests']+['—']*9
    latex_file(output/'category_by_scene_condition.tex',category_headers,[missing]+category_table,'Object reconstruction on scene-derived crops, grouped by furniture category. Different rows use different held-out subsets.')
    figure_file(output/'category_by_scene_condition.png',['Training/test condition']+['CD ↓','Seg Acc\n(%) ↑','Seg mIoU\n(%) ↑']*3,
                [missing]+category_table,'Current object reconstruction results by furniture category',
                'Original isolated-object tests have no measured results yet. Rows are matched scene-size training/test conditions, not a common-test ranking. Segmentation uses available annotations only; see CSV for coverage. Cabinet / storage is the code category storagefurniture. Values are measured project results, not values copied from the reference paper.',
                [(display,1+i*3,3+i*3) for i,(_,display) in enumerate(CATEGORIES)])
    csv_file(output/'scene_size_comparison.csv',size_rows,list(size_rows[0]))
    size_headers=['Training/test condition','Objects','CD ↓','Seg Acc\n(%) ↑','Seg mIoU\n(%) ↑','Seg.\ncoverage','CAD success\n(%)']
    size_table=[[r['condition'],r['test_objects'],number(r['cd']),number(r['seg_accuracy_pct']/100 if r['seg_accuracy_pct'] is not None else None,True),number(r['seg_miou_pct']/100 if r['seg_miou_pct'] is not None else None,True),f"{r['segmentation_objects']}/{r['test_objects']}",f"{r['generation_success_pct']:.1f}"] for r in size_rows]
    latex_file(output/'scene_size_comparison.tex',size_headers,size_table,'Object-level metrics by scene-size condition. Matched tests differ between rows.')
    figure_file(output/'scene_size_comparison.png',size_headers,size_table,'Current object reconstruction results by scene-size condition',
                'These are per-object metrics. Specialized models use their matching test size; the combined model uses all 50 test crops. Test subsets and category composition differ. Missing segmentation annotations are excluded, not counted as zero. CAD export success does not establish accurate geometry.')
    scene_rows=[]
    scene_root=ROOT/'data/output/scene_assemblies/combined/test'
    for path in scene_root.glob('*/metrics.json'):
        scene=json.loads((path.parent/'scene.json').read_text());metric=json.loads(path.read_text())
        scene_rows.append(dict(scene_id=scene['scene_id'],object_count=scene['expected_objects'],status=scene['status'],object_coverage=scene['object_coverage'],scene_cd=metric.get('scene_cd_gt_diagonal'),placement=scene['placement']))
    if scene_rows:
        csv_file(output/'whole_scene_per_scene.csv',scene_rows,list(scene_rows[0]))
        summary=[]
        for count in (1,2,3,4,None):
            selected=[r for r in scene_rows if count is None or r['object_count']==count]
            if not selected:continue
            values=[r['scene_cd'] for r in selected if r['scene_cd'] is not None]
            summary.append(dict(condition='All scenes' if count is None else f'{count}-object scenes',scenes=len(selected),complete=sum(r['status']=='complete' for r in selected),object_coverage_pct=100*sum(r['object_coverage'] for r in selected)/len(selected),mean_scene_cd=sum(values)/len(values) if values else None,placement='ground_truth'))
        csv_file(output/'whole_scene_comparison.csv',summary,list(summary[0]))
        headers=['Scene group','Scenes','Complete','Object coverage\n(%)','Scene CD ↓','Placement']
        rows=[[r['condition'],r['scenes'],r['complete'],f"{r['object_coverage_pct']:.1f}",number(r['mean_scene_cd']),'Annotated'] for r in summary]
        latex_file(output/'whole_scene_comparison.tex',headers,rows,'Whole-scene CAD with annotated placement. Scene CD is a separate metric from object CD.')
        figure_file(output/'whole_scene_comparison.png',headers,rows,'Whole-scene assembly results — combined geometry model',
                    'Scene CD uses the shared scene frame divided by the reference scene bounding-box diagonal. It is not comparable numerically to independently normalized object CD. Lower multi-object scores alone do not demonstrate better reconstruction. Placement is annotated, not learned.')
    # A controlled comparison: specialist routing and shared model on the same 50 crops.
    metric_root=ROOT/'data/output/trassembler_h5/scene'
    def read_rows(group):
        path=metric_root/group/'test/metrics_per_object.jsonl'
        return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    shared=read_rows('all_objects')
    specialists=[row for group,_ in CONDITIONS if group!='all_objects' for row in read_rows(group)]
    assert len({r['sample_id'] for r in shared})==len(shared), 'Duplicate shared-model samples'
    assert len({r['sample_id'] for r in specialists})==len(specialists), 'Duplicate specialist samples'
    assert {r['sample_id'] for r in shared}=={r['sample_id'] for r in specialists}, 'Cannot compare policies on different test IDs'
    assert {r['sample_id'] for r in shared if r.get('partseg_accuracy') is not None} == {r['sample_id'] for r in specialists if r.get('partseg_accuracy') is not None}, 'Segmentation policies have different scored subsets; do not present this as a matched comparison'
    controlled=[];controlled_table=[]
    for label,items in [('Scene-size specialists',specialists),('Shared all-scenes model',shared)]:
        row=[label]
        for category,display in CATEGORIES:
            selected=[r for r in items if r['category']==category]
            cd=[r['cd'] for r in selected if r.get('cd') is not None]
            seg=[r for r in selected if r.get('partseg_accuracy') is not None]
            record=dict(policy=label,category=category,test_objects=len(selected),cd_objects=len(cd),
                        cd=sum(cd)/len(cd) if cd else None,
                        segmentation_objects=len(seg),
                        seg_accuracy_pct=100*sum(r['partseg_accuracy'] for r in seg)/len(seg) if seg else None,
                        seg_miou_pct=100*sum(r['partseg_instance_avg_iou'] for r in seg)/len(seg) if seg else None)
            controlled.append(record)
            row += [number(record['cd']),number(record['seg_accuracy_pct']/100 if seg else None,True),number(record['seg_miou_pct']/100 if seg else None,True)]
        controlled_table.append(row)
    csv_file(output/'category_comparison.csv',controlled,list(controlled[0]))
    latex_file(output/'category_comparison.tex',category_headers,controlled_table,
               'Shared model versus scene-size specialist routing on the same held-out scene crops. Segmentation uses available annotations only.')
    figure_file(output/'category_comparison.png',['Reconstruction policy']+['CD ↓','Seg Acc\n(%) ↑','Seg mIoU\n(%) ↑']*3,
                controlled_table,'Scene-crop reconstruction: shared model versus scene-size specialists',
                'Both policies use the same 50 test crops. Specialists are routed using the supplied source-scene object count. Segmentation uses only annotated objects; coverage is in CSV. These are scene-crop tests, not the original isolated-object test dataset. Cabinet / storage denotes storagefurniture. No published benchmark values are copied.',
                [(display,1+i*3,3+i*3) for i,(_,display) in enumerate(CATEGORIES)])
    matched=[];matched_table=[]
    for group,label in CONDITIONS:
        if group=='all_objects':continue
        count=int(group.split('_')[0])
        a=[r for r in specialists if r['object_count']==count]
        b=[r for r in shared if r['object_count']==count]
        assert {r['sample_id'] for r in a}=={r['sample_id'] for r in b}
        item=dict(test_scene_size=count,test_objects=len(a))
        for prefix,items in [('specialist',a),('shared',b)]:
            cd=[r['cd'] for r in items if r.get('cd') is not None]
            seg=[r for r in items if r.get('partseg_accuracy') is not None]
            item.update({prefix+'_cd':sum(cd)/len(cd) if cd else None,
                         prefix+'_seg_acc_pct':100*sum(r['partseg_accuracy'] for r in seg)/len(seg) if seg else None,
                         prefix+'_seg_miou_pct':100*sum(r['partseg_instance_avg_iou'] for r in seg)/len(seg) if seg else None,
                         prefix+'_seg_objects':len(seg)})
        matched.append(item)
        matched_table.append([f'{count}-object scenes',len(a),number(item['specialist_cd']),
                              number(item['specialist_seg_acc_pct']/100 if item['specialist_seg_acc_pct'] is not None else None,True),number(item['specialist_seg_miou_pct']/100 if item['specialist_seg_miou_pct'] is not None else None,True),
                              number(item['shared_cd']),number(item['shared_seg_acc_pct']/100 if item['shared_seg_acc_pct'] is not None else None,True),number(item['shared_seg_miou_pct']/100 if item['shared_seg_miou_pct'] is not None else None,True),
                              f"{item['specialist_seg_objects']}/{len(a)} | {item['shared_seg_objects']}/{len(b)}"])
    csv_file(output/'matched_test_comparison.csv',matched,list(matched[0]))
    matched_headers=['Test subset','Objects','CD ↓','Seg Acc\n(%) ↑','Seg mIoU\n(%) ↑','CD ↓','Seg Acc\n(%) ↑','Seg mIoU\n(%) ↑','Seg.\ncoverage']
    latex_file(output/'matched_test_comparison.tex',['Test subset','Objects','Specialist CD','Specialist Acc (%)','Specialist mIoU (%)','Shared CD','Shared Acc (%)','Shared mIoU (%)','Coverage'],matched_table,
               'Specialist versus shared model on identical held-out objects within each scene-size group.')
    figure_file(output/'matched_test_comparison.png',matched_headers,matched_table,
                'Controlled comparison within each held-out scene-size group',
                'Within each row, both policies evaluate the same test objects. Different rows contain different objects/categories. Coverage is shown specialist | shared. Lower CD and higher segmentation scores are better; these are per-object metrics, not full-scene placement metrics.',
                [('Scene-size specialist',2,4),('Shared all-scenes model',5,7)])
    # Original-trained baseline: same scene inputs, with missing runs left explicit.
    baseline_root=ROOT/'data/output/baselines'
    baseline_rows=[];baseline_table=[];matches=[]
    for domain,label in [('scenes','Original models / scene crops'),('original','Original models / isolated tests')]:
        row=[label]
        for category,display in CATEGORIES:
            folder=baseline_root/f'original_on_{domain}'/'trassembler'/category
            if domain=='scenes':folder=folder/'test'
            path=folder/'metrics.json'
            report=json.loads(path.read_text()) if path.is_file() else {}
            metric=report.get('overall',{})
            baseline_rows.append(dict(training='original',test_domain=domain,category=category,
                status='measured' if report else 'pending',test_objects=metric.get('total_samples'),
                cd=metric.get('mean_cd'),seg_accuracy_pct=None if metric.get('partseg_accuracy') is None else 100*metric['partseg_accuracy'],
                seg_miou_pct=None if metric.get('partseg_instance_avg_iou') is None else 100*metric['partseg_instance_avg_iou'],
                segmentation_objects=metric.get('partseg_sample_count'),source=str(path)))
            row += [number(metric.get('mean_cd')),number(metric.get('partseg_accuracy'),True),number(metric.get('partseg_instance_avg_iou'),True)]
            per_object=folder/'metrics_per_object.jsonl'
            indexed={r['sample_id']:r for r in (json.loads(line) for line in per_object.read_text().splitlines() if line.strip())} if per_object.is_file() else {}
            selected=[r for r in shared if r['category']==category]
            if domain=='scenes' and indexed:
                assert set(indexed)=={r['sample_id'] for r in selected}, 'Original/scene baseline has different test IDs'
            for item in selected:
                key=item['sample_id'] if domain=='scenes' else str(item['reference_id'])
                baseline=indexed.get(key,{})
                matches.append(dict(scene_sample_id=item['sample_id'],scene_id=item['scene_id'],reference_id=item['reference_id'],
                    category=category,object_count=item['object_count'],baseline_test_domain=domain,
                    baseline_sample_id=key,shared_scene_cd=item.get('cd'),original_model_cd=baseline.get('cd'),
                    shared_seg_accuracy=item.get('partseg_accuracy'),original_seg_accuracy=baseline.get('partseg_accuracy'),
                    shared_seg_miou=item.get('partseg_instance_avg_iou'),original_seg_miou=baseline.get('partseg_instance_avg_iou')))
        baseline_table.append(row)
    shared_table=next(r for r in controlled_table if r[0]=='Shared all-scenes model')
    headers=['Training / test domain']+['CD ↓','Seg Acc\n(%) ↑','Seg mIoU\n(%) ↑']*3
    comparison_rows=baseline_rows+[dict(training='scenes_shared',test_domain='scenes',category=r['category'],
        status='measured',test_objects=r['test_objects'],cd=r['cd'],seg_accuracy_pct=r['seg_accuracy_pct'],
        seg_miou_pct=r['seg_miou_pct'],segmentation_objects=r['segmentation_objects'],
        source=str(metric_root/'all_objects/test/metrics.json')) for r in controlled if r['policy']=='Shared all-scenes model']
    csv_file(output/'original_model_comparison.csv',comparison_rows,list(baseline_rows[0]))
    csv_file(output/'comparison_matches.csv',matches,list(matches[0]))
    latex_file(output/'original_model_comparison.tex',category_headers,baseline_table+[shared_table],
        'Original-trained models and scene-trained model. Original models on scene crops use the same test IDs; isolated tests are a separate domain.')
    figure_file(output/'original_model_comparison.png',headers,baseline_table+[shared_table],
        'Original-trained versus scene-trained reconstruction',
        'Dashes mean inference/evaluation has not been run. Compare scene-crop rows on identical images; isolated tests use a separate image domain and test population. Existing scene outputs lack inference protocol metadata, so final protocol equivalence still needs verification. Coverage and raw sample matching are in CSV.',
        [(display,1+i*3,3+i*3) for i,(_,display) in enumerate(CATEGORIES)])
    assembly_comparison=[]
    for mode in ('combined','original','specialized'):
        for path in sorted((ROOT/'data/output/scene_assemblies'/mode/'test').glob('*/metrics.json')):
            scene=json.loads((path.parent/'scene.json').read_text());metric=json.loads(path.read_text())
            assembly_comparison.append(dict(model=mode,scene_id=scene['scene_id'],object_count=scene['expected_objects'],
                status=scene['status'],scene_cd=metric.get('scene_cd_gt_diagonal'),placement=scene['placement']))
    if assembly_comparison:
        csv_file(output/'whole_scene_model_comparison.csv',assembly_comparison,list(assembly_comparison[0]))
    training=[]
    for path in sorted((ROOT/'logs').glob('trassembler-*.log')):
        text=path.read_text(errors='replace')
        experiment=re.search(r'Experiment directory: (.+)',text)
        epochs=re.findall(r'Epoch\s+(\d+)',text)
        if experiment and 'Training completed!' in text:
            training.append(dict(run=experiment.group(1),completed=True,epochs=int(epochs[-1])+1 if epochs else None,log=str(path)))
    missing_protocol=0
    for group,_ in CONDITIONS:
        for row in read_rows(group):
            path=metric_root/group/'test'/row['sample_id']/'metadata.json'
            metadata=json.loads(path.read_text()) if path.is_file() else {}
            missing_protocol += not isinstance(metadata.get('inference_options'),dict)
    manifest=dict(original_model_baselines=baseline_rows,sources=sources,missing_original_results=status,placement_status=placement,
                  completed_training_runs=training,matched_test_ids=True,matched_segmentation_ids=True,
                  prediction_records_missing_protocol_metadata=int(missing_protocol),
                  limitations=['Segmentation annotations cover only a subset of test objects.',
                               'Matched scene-size rows use different test sets; train/test crossing is not complete.',
                               'Whole-scene and per-object Chamfer definitions differ.',
                               'Current reconstructions have visible geometry errors despite successful exports.',
                               'Legacy prediction metadata does not record inference options; verify/refine protocol before final paper comparisons.',
                               'Paper-image benchmark values are not imported or claimed as project results.'])
    (output/'comparison_status.json').write_text(json.dumps(manifest,indent=2)+'\n')
    print(f'Comparison tables saved to {output}')
    print(json.dumps(placement))


if __name__=='__main__':
    main()
