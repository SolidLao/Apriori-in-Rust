//! Implement Apriori Algorithm in Rust
//! 
//! how to use this algorithm:
//! 
//! ```
//! // set min_sup and min_conf
//! let min_sup = 0.005;
//! let min_conf = 0.3;
//!
//! // call the apriori function
//! let (_fre_sets, association_rules_set) = apriori(min_sup, min_conf, "groceries.csv");
//! 
//! // write all association rules to file
//! write_rules_to_file("associationRule.txt", &association_rules_set);
//! ```

use std::{collections::HashMap, time::SystemTime, io::Write, mem::size_of_val};

/// # transaction consists of items
/// 
/// id: the id of the Txn
/// 
/// items: a Vec that contains the items in this transaction
#[derive(Debug)]
#[allow(dead_code)]
pub struct Txn {
    id: usize,
    items: Vec<String>,
}

/// # to be selected as FrequentSet
/// 
/// degree: how many items it has
/// 
/// items: the Vec of String, containing all the items
/// 
/// count: the times the set occurs in all the transactions
#[derive(Debug)]
pub struct CandicateSet {
    degree: usize,
    items: Vec<String>,
    count: usize,
}

/// # CandicateSet whose count is greater than (txn_count * min_sup)
/// 
/// it contains:
/// 
/// degree: how many items it has
/// 
/// items: the Vec of String, containing all the items
/// 
/// count: the times the set occurs in all the transactions
#[derive(Clone, Debug)]
pub struct FrequentSet {
    degree: usize,
    items: Vec<String>,
    count: usize,
}

/// # the final rules we want
/// 
/// from -> to
/// 
/// with its support and confidence
#[derive(Debug)]
pub struct AssociationRule {
    from: Vec<String>,
    to: Vec<String>,
    sup: f64,
    conf: f64,
}

/// the ultimate interface to call apriori function
/// 
/// arg
/// min_sup: minimum support
/// min_conf: minimum confidence
/// filename: the dataset's filename, only for csv file now
/// 
/// return 
/// fre_sets: all frequentSet
/// association_rule_set: all association rules
pub fn apriori(min_sup: f64, min_conf: f64, filename: &str) -> (Vec<FrequentSet>, Vec<AssociationRule>) {

    // init
    // the set of all frequent set, 'sets' means the set of set
    let mut fre_sets: Vec<FrequentSet> = Vec::new();
    // generate association rules from fre_sets
    let mut association_rules_set: Vec<AssociationRule> = Vec::new();

    // get all transactions from file
    let mut txn_set = create_sorted_txn_set(filename);    

    // generate 1-CandicateSet and thus 1-FrequentSet and add it in to the frequent sets
    init_fre_set(&mut txn_set, min_sup, &mut fre_sets);
    
    // the core of the Apriori Algorithm: find frequentSet of all degrees
    // generate all FrequentSets from 1-FrequentSet
    generate_all_fre_sets(&mut fre_sets, &txn_set, min_sup);
    
    // find all association rules
    generate_association_rules(&fre_sets, min_conf, &mut association_rules_set, txn_set.len());

    // repoart space consumption
    let fre_size: usize = fre_sets.iter().map(|x| size_of_val(&x)).sum();
    let rules_size: usize  = association_rules_set.iter().map(|x| size_of_val(&x)).sum();

    println!("\nSpace Consumption");
    println!("The size, len of fre_sets: {} bytes, {}", fre_size, fre_sets.len());
    println!("The size, len of association_rules_set: {} bytes, {}", rules_size, association_rules_set.len());

    return (fre_sets, association_rules_set)
}

/// write all association rules to file
pub fn write_rules_to_file(filename: &str, association_rules_set: &Vec<AssociationRule>) {
    
    let mut file = std::fs::File::create(filename).expect("failed to create associationRule.txt");

    for (i, rule) in association_rules_set.iter().enumerate() {
        file.write_fmt(format_args!("\nrule {}:\n{:#?} --> {:#?}, sup = {}, conf = {}\n", 
            i, rule.from, rule.to, rule.sup, rule.conf)).expect("failed to write rules to file");
    }
}

/// generate all association rules
fn generate_association_rules(fre_sets: &Vec<FrequentSet>, min_conf: f64, association_rules_set: &mut Vec<AssociationRule>, txn_num: usize) {

    // start
    println!("\nStarting to find all Association Rules **********************************************");
    println!("min_conf: {}", min_conf);
    let rule_start_time = SystemTime::now();

    // iterate over non-empty real subset of each FrequentSet
    for fre_set in fre_sets.iter() {

        let degree = fre_set.degree;
        
        // freset whose degree is 1, is ignored
        if degree == 1 {
            continue;
        }

        // iterate over all subsets of fre_set.items, 2^n -1 in total
        for mut i in 1..((2 as usize).pow(degree as u32) - 1) {

            // association rule: from -> to
            let mut from: Vec<String> = Vec::new();
            let mut pos = 0;

            while i != 0 {

                // bitmap: 1 for from
                if i&1 == 1 {
                    from.push(fre_set.items[pos].clone());
                } 

                pos += 1;
                i >>= 1;
            }

            // the rest is for to
            let to: Vec<String> = fre_set.items.iter().filter(|&x| !from.contains(x)).map(|x| x.to_string()).collect::<Vec<String>>().to_vec();

            // calculate conf for the rule
            let from_fre_set = fre_sets.iter().find(|&x| x.items.eq(&from)).unwrap();
            let conf = fre_set.count as f64 / from_fre_set.count as f64;

            // if conf >= min_conf, this rule is an association rule!
            if conf >= min_conf {
                let new_rule = AssociationRule {
                    from,
                    to,
                    sup: fre_set.count as f64 / txn_num as f64,
                    conf,
                };

                association_rules_set.push(new_rule);
            }
        }
    }

    // end
    let rule_finish_time = SystemTime::now();
    println!("Finished to find all Association Rules **********************************************");
    println!("It takes {:#?} to find all frequentSets", rule_finish_time.duration_since(rule_start_time).unwrap());
}

/// generate all FrequentSets from 1-FrequentSet
fn generate_all_fre_sets(fre_sets: &mut Vec<FrequentSet>, txn_set: &Vec<Txn>, min_sup: f64) {

    // start
    let fre_start_time = SystemTime::now();
    println!("\nStarting to find all FrequentSet **********************************************");
    println!("min_sup: {}, min_count for support: {}\n", min_sup, (txn_set.len() as f64 * min_sup) as usize);

    // calculate min_count from min_sup
    let min_count = (txn_set.len() as f64 * min_sup) as usize;

    // calculate the len of FrequentSet based on degree
    let mut degree = 1;
    let mut len_of_f = len_of_f_degree(&fre_sets, degree);

    // when f of degree is empty, the loop is over
    while len_of_f > 0 {
        println!("degree: {}, num of corresponding frequentSet: {}", degree, len_of_f);

        // candi_sets.count = 0 at this moment
        // len of set in candi_sets is degree + 1
        let candi_sets: Vec<CandicateSet> = get_candi_from_f(&fre_sets, degree);

        for mut candi_set in candi_sets {
            
            for txn in txn_set.iter() {

                // if candi_set.items is subset of txn.items
                // candi_set.count += 1
                if subset_of(&candi_set.items, &txn.items) {
                    candi_set.count += 1;
                }
            }

            // if candi_set.items >= min_sup
            // convert it to FrequentSet and add it to fre_sets
            if candi_set.count >= min_count {
                let new_fre = FrequentSet {
                    degree: candi_set.degree,
                    items: candi_set.items,
                    count: candi_set.count,
                };

                fre_sets.push(new_fre);
            }
        }

        degree += 1;
        len_of_f = len_of_f_degree(&fre_sets, degree);

    }

    // end
    let fre_finish_time = SystemTime::now();
    println!("\nFinished to find all FrequentSet **********************************************");
    println!("It takes {:#?} to find all frequentSets", fre_finish_time.duration_since(fre_start_time).unwrap());

}

/// generate 1-CandicateSet and thus 1-FrequentSet and add it in to the frequent sets
fn init_fre_set(txn_set: &mut Vec<Txn>, min_sup: f64, fre_sets: &mut Vec<FrequentSet>) {
    // generate C_1
    let candicate_set_1 = create_candicate_set_1(&txn_set);
    // generate F_1
    let min_count = (txn_set.len() as f64 * min_sup) as usize;
    let frequent_set_1 = create_frequent_set_1(candicate_set_1, min_count);

    // add F_1 to fre_set
    for set in frequent_set_1.iter() {
        let fre_set = FrequentSet {
            degree: 1,
            items: vec![set.0.clone()],
            count: set.1.clone(),
        };

        fre_sets.push(fre_set);
    }
}

/// judge whether a set is a subset of another set
/// 
/// this fn can be expanded to generics in the future
fn subset_of(subset: &Vec<String>, set: &Vec<String>) -> bool {

    for item in subset {
        if !set.contains(item) {
            return false;
        }
    }

    true
}

/// generate set of K-CandicateSet from set of (K-1)-FrequentSet
/// 
/// if two (K-1)-FrequentSets, the first K-2 elements are identical and the (k-1)th are different
/// 
/// then generate new CandicateSet with (degree - 1) elements and the degree-th element 
fn get_candi_from_f(fre_sets: &Vec<FrequentSet>, degree: usize) -> Vec<CandicateSet> {

    let mut candi:Vec<CandicateSet> = Vec::new();

    // fre_sets is a set of frequent_set with all kinds of degrees
    // get degree-frequent_sets which is a set of frequent_set with degree items
    let degree_fre_sets = get_degree_fre_sets(fre_sets, degree);

    for i in 0..(degree_fre_sets.len() - 1) {
        for j in i+1..degree_fre_sets.len() {

            // the first (degree -1) elements are identical
            if degree_fre_sets[i].items.as_slice()[0..(degree-1)] == degree_fre_sets[j].items.as_slice()[0..(degree-1)] {

                // the degree-th elements are different
                if degree_fre_sets[i].items.get(degree - 1).unwrap() != degree_fre_sets[j].items.get(degree - 1).unwrap() {

                    // then generate new CandicateSet with (degree - 1) elements and the degree-th element
                    // generate items
                    let mut items = degree_fre_sets[i].items.clone();
                    items.push(degree_fre_sets[j].items.get(degree - 1).unwrap().clone());

                    let count = 0;

                    let new_candi = CandicateSet {
                        degree: degree + 1,
                        items,
                        count,
                    };

                    candi.push(new_candi);
                }
            }
        }
    }

    candi
}

/// fre_sets is a set of frequent_set with all kinds of degrees
/// 
/// get degree-frequent_sets which is a set of frequent_set with degree items
fn get_degree_fre_sets(fre_sets: &Vec<FrequentSet>, degree: usize) -> Vec<FrequentSet> {
    fre_sets.clone().into_iter().filter(|x| x.degree == degree).collect::<Vec<FrequentSet>>().to_vec()
}

/// ## get len of f based on the degree
/// 
/// the fre_sets is a set of all FrequentSet, degree indicates the number of item in each FrequentSet
fn len_of_f_degree(fre_sets: &Vec<FrequentSet> , degree: usize) -> usize {
    fre_sets.iter().filter(|x| x.degree == degree).count() as usize
}

/// ## generate frequent_set_1, given clone of candicate_set_1 and min_count
/// 
/// if candicate_set's count is larger than len(txn_set) * min_sup, the set is frequent
/// 
/// to avoid changing the value in candicate_set_1, use clone of it
/// 
/// min_count is calculated by multiplying length of txn_set and min_sup
fn create_frequent_set_1(candicate_set_1: HashMap<String, usize>, min_count: usize) -> HashMap<String, usize> {
    let frequent_set_1: HashMap<String, usize> 
        = candicate_set_1.into_iter()
            .filter(|x| x.1.clone()  >  min_count)
            .collect::<HashMap<_,_>>();

    frequent_set_1
}

/// ## generate candicate_set_1 from txn_set
/// 
/// candicate_set_1 is a hashmap which contains only one item(String) and its corresponding count
/// 
/// txn_set: Vec of Txn
/// 
/// By using hashMap, we can create candicate_set_1 conveniently
fn create_candicate_set_1(txn_set: &Vec<Txn>) -> HashMap<String, usize> {
    
    // generate C_1
    let mut candicate_set_1: HashMap<String, usize> = HashMap::new();
    for txn in txn_set.iter() {

        for item in txn.items.clone() {
            candicate_set_1.entry(item).and_modify(|x| *x += 1).or_insert(1);
        }
    }

    candicate_set_1
}

/// ## generate txn_set from csv file, the items in each txn are sorted in lexicographic order
/// 
/// filename: the path and name of the dataset.csv
fn create_sorted_txn_set(filename: &str) -> Vec<Txn> {

    let mut txn_set: Vec<Txn> = Vec::new();

    let mut reader = csv::Reader::from_path(filename).expect("failed to read the csv file");

    for (i, items_result) in reader.records().enumerate() {
        let items = items_result.expect("faile to get items from txn_result");

        let mut items_vec: Vec<String> = items.iter().filter(|x| !x.is_empty()).map(|x| x.to_string()).collect();

        // sort the items for each txn in txn_set in lexicographic order
        items_vec.sort();

        let txn = Txn {
            id: i,
            items: items_vec,
        };
        
        txn_set.push(txn);
    }

    txn_set
}

/// get good filename based on min_sup and min_conf
pub fn get_good_filename(min_sup: f64, min_conf: f64) -> String {
    let mut filename: String = "associationRule_".to_string();
    filename += &min_sup.to_string();
    filename += "_";
    filename += &min_conf.to_string();
    filename += ".txt";

    filename
}